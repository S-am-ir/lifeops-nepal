from typing import Literal
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.config.settings import settings
from src.agents.state import AgentState, IntentClassification


def get_classifier_llm():
    """Gemini Flash for classification - flash and frugal"""
    if settings.google_api_key:
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=settings.google_api_key.get_secret_value(),
            temperature=0,
        )
    # Fallback to Groq if Google key not set
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=settings.groq_token.get_secret_value() if settings.groq_token else None,
    )

# System Prompt

CLASSIFIER_SYSTEM = """You are an intent classifier for a personal life admin AI.

Classify the latest user message into ONE of:

travel_planning — planning or researching trips, flights, hotels, weather, destination info,
  packing advice, itineraries, "what to do in X", budget for travel.
  Examples: "Plan a trip to Pokhara", "Flights KTM to DEL next Friday",
            "What should I do in Kathmandu for 2 days?", "Is it rainy in Pokhara in March?"

reminder — user wants to be reminded, notified, or alerted at some time via WhatsApp.
  Examples: "Remind me to call mom at 5pm", "Alert me tomorrow morning",
            "Send me a message in 2 hours"

creative — generating images, moodboards, visual concepts, aesthetic exploration.
  Examples: "Generate a moodboard for a mountain trip", "Create an image of Pokhara at sunset"

unknown — anything that doesn't fit the above, or is a greeting/meta question.
  Examples: "Hi", "What can you do?", "Thanks"

IMPORTANT: Look at the FULL conversation history for context. A short follow-up like
"yes go ahead" or "what about hotels?" belongs to the same intent as the prior messages.

Respond with valid JSON only:
{
    "intent": "travel_planning",
    "confidence": 0.95,
    "reasoning": "User is asking about flights"
}
"""

async def classify_intent_node(state: AgentState) -> dict:
    """Classify user intent from latest message + conversation context."""

    messages = state.get("messages", [])
    if not messages:
        return {"intent": "unknown"}
    
    llm = get_classifier_llm()
    structured_llm = llm.with_structured_output(IntentClassification)

    try:
        result: IntentClassification = await structured_llm.ainvoke(
            [SystemMessage(content=CLASSIFIER_SYSTEM)] + messages
        )
        intent = result.intent
        print(f"[Orchestrator] Intent: {intent} ({result.confidence:.0%})")
    except Exception as e:
        print(f"[Orchestrator] Classification failed: {e}")
        intent = "unknown"
    
    return {"intent": intent}

# Router
def route_to_agent(state: AgentState) -> Literal["travel_agent", "reminder_agent", "creative_agent", "unknown_handler"]:
    """Conditional edge: route based on classified intent"""
    return {
        "travel_planning": "travel_agent",
        "reminder": "reminder_agent",
        "creative": "creative_agent",
        "unknown": "unknown_handler"
    }.get(state.get("intent", "unknown"), "unknown_handler")

# Unknown handler
async def unknown_handler_node(state: AgentState) -> dict:
    """Friendly fallback for unrecognized queries"""

    response = (
        "I'm your personal life admin assistant. Here's what I can help with:\n\n"
        "✈️  **Travel** — flights, hotels, weather, destination tips, itineraries\n"
        "🔔  **Reminders** — send yourself a WhatsApp reminder at any time\n"
        "🎨  **Creative** — AI moodboards and image generation\n\n"
        "What would you like to do?"
    )

    return {
        "message": [AIMessage(content=response)],
        "final_response": response,
    }