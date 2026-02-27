import json
from datetime import datetime, timezone
from typing import Optional
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from langchain_core.messages import SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from src.agents.state import AgentState, ReminderExtraction
from src.mcp.client import get_mcp_tools
from src.config.settings import settings

# Shared APScheduler instance
# Initialized lazily on first use on it doesn't block app startup.

_scheduler = None

def _get_scheduler():
    global _scheduler
    if _scheduler is None:
        _scheduler = AsyncIOScheduler()
        _scheduler.start()
        print("[Reminder] APScheduler started")
    return _scheduler

# LLM
def get_reminder_llm():
    if settings.google_api_key:
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=settings.google_api_key.get_secret_value(),
            temperature=0, 
        )
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=settings.groq_token.get_secret_value() if settings.groq_token else None
    )

# System Prompt
REMINDER_SYSTEM = """ You are extracting reminder details from a user's message.

Today (Nepal time): {today}

Extract:
- reminder_message: concise WhatsApp-ready text of what to remind about
- scheduled_for: ISO datetime (YYYY-MM-DDTHH:MM:SS) if a specific time was given,
or "now" for immediate send (also treat "within", "until" as immediate e.g. within 2 minutes).Nepal is UTC+5:45.
- to_number: phone number if user specified one (international format not + ), else null
- repeat_rule: "daily", "weekly", or "none"

Respond with valid JSON only
"""

# Node
async def reminder_agent_node(state: AgentState) -> dict:
    """Extract reminder details, schedule or send immediately"""

    messages = state.get("messages", [])
    llm = get_reminder_llm()
    structured_llm = llm.get_structured_output(ReminderExtraction)

    now_iso = datetime.now().strftime("%Y-%m-%d %H:%M")

    try:
        extracted: ReminderExtraction = await structured_llm.ainvoke(
            [SystemMessage(content=REMINDER_SYSTEM.format(today=now_iso))] + messages
        )
    except Exception as e:
        err = f"Couldn't parse your reminder. Please try again with a clearer time. {str(e)}"
        return {
            "messages": [AIMessage(content=err)],
            "final_response": err,
        }
    
    # Resolve phone number
    to_number = extracted.to_number or state.get("user_phone")
    if not to_number:
        msg = (
            "I couln't find a phone number to send the reminder to."
            "Please share your WhatsApp number (e.g. 9779812345678)."
        )
        return {"messages": [AIMessage(content=msg)], "final_response": msg}
    
    reminder_text = extracted.reminder_message
    scheduled_for = extracted.scheduled_for

    send_now = scheduled_for.lower() == "now"

    if not send_now:
        try:
            run_dt = datetime.fromisoformat(scheduled_for)
        except ValueError:
            send_now = True # unparseable datetime → send now as safe fallback

    if send_now:
        response_text = await _send_whatsapp_now(to_number, reminder_text)
    else:
        response_text = await _schedule_reminder(
            to_number=to_number,
            message=reminder_text,
            run_dt=run_dt,
            repeat_rule=extracted.repeat_rule or "none",
        )
    
    return {
        "message": [AIMessage(content=response_text)],
        "final_response": response_text,
    }

# Helpers
async def _send_whatsapp_now(to_number: str, message: str) -> str:
    """Fire the WhatsApp message immediately via MCP comms tool"""
    tools = await get_mcp_tools(servers=["comms"])
    whatsapp_tool = next((t for t in tools if t.name == "send_whatsapp_message"), None)

    if not whatsapp_tool:
        return "⚠️ WhatsApp tool not available. Is the comms MCP server running?"
    
    try:
        result = await whatsapp_tool.ainvoke({"to_number": to_number, "body": message})
        result_dict = json.loads(result) if isinstance(result, str) else result

        if result_dict.get("status") == "sent":
            return f"✅ Reminder sent to {to_number}!\n\n📱 Message: _{message}_"
        
        else:
            err = result_dict.get("error", "Unknow error occured")
            return f"❌ Failed to send reminder: {err}"
    except Exception as e:
        return f"❌ WhatsApp send failed: {e}"
    
async def _schedule_reminder(to_number: str, message: str, run_dt: datetime, repeat_rule: str) -> str:
    """Schedule a future WhatsApp send via APScheduler"""

    scheduler = _get_scheduler()

    job_kwargs = dict(
        func=_fire_reminder,
        args=[to_number, message],
        id=f"reminder_{to_number}_{run_dt.timestamp():.0f}",
        replace_existing=True,
    )

    if repeat_rule == "daily":
        from apscheduler.triggers.cron import CronTrigger
        job_kwargs["trigger"] = CronTrigger(
            hour=run_dt.hour, minute=run_dt.minute
        )
        repeat_label = "daily"
    elif repeat_rule == "weekly":
        from apscheduler.triggers.cron import CronTrigger
        job_kwargs["trigger"] = CronTrigger(
            day_of_week=run_dt.strftime("%a").lower(),
            hour=run_dt.hour,
            minute=run_dt.minute,
        )
        repeat_label = f"every {run_dt.strftime('%A')}"
    else:
        from apscheduler.triggers.date import DateTrigger
        job_kwargs["trigger"] = DateTrigger(run_date=run_dt)
        repeat_label = "once"

    scheduler.add_job(**job_kwargs)

    friendly_time = run_dt.strftime("%A, %d %b %Y at %I:%M %P")
    return (
        f"⏰ Reminder scheduled ({repeat_label})!\n\n"
        f"📅 Time: {friendly_time}\n"
        f"📱 To: {to_number}\n"
        f"💬 Message: _{message}_"
    )

async def _fire_reminder(to_number: str, message: str):
    """APScheduler callback - fires the actual WhatsApp send"""
    print(f"[Reminder] Firing scheduled reminder to {to_number}")
    await _send_whatsapp_now(to_number, message)