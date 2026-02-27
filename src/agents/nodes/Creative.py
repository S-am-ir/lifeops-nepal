import json
from langchain_core.messages import SystemMessage, AIMessage
from src.agents.state import AgentState, CreativeExtraction
from src.mcp.client import get_mcp_tools
from src.config.settings import settings

# LLM
def get_creative_llm():
    if settings.google_api_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=settings.google_api_key.get_secret_value() if settings.google_api_key else None,
            temperature=0.4, # slightly creative for prompt expansion
        )
    from langchain_groq import ChatGroq
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        api_key=settings.groq_token.get_secret_value() if settings.groq_token else None,
    )

# System Prompt

CREATIVE_SYSTEM = """You are a visual prompt engineer for AI image generation

Expand the user's creative request into a richly descriptive visual prompt
suitable for a diffusion model (fal.ai / FLUX).

Include: lightning, mood, colour palette, setting, time of day, style references,
camera language (wide shot, bokeh, cinematic, etc.)

Nepal context: if the subject involves Nepal, weave into authentic elements -
Himalayan peaks, prayer flags, terraced fields, pagodas, local markets etc.

Keep `visual_prompt` under 120 words. Set `count` to 2 if the user wants variety,
else 1.

Respond with JSON only.
"""

# Node

async def creative_agent_node(state: AgentState) -> dict:
    """Expand prompt -> generate moodboard -> respond with image URLs"""

    messages = state.get("messages", [])
    llm = get_creative_llm()
    structured_llm = llm.with_structured_output(CreativeExtraction)

    try:
        extracted: CreativeExtraction = await structured_llm.ainvoke(
            [SystemMessage(content=CREATIVE_SYSTEM)] + messages
        )
    except Exception as e:
        err = f"Couldn't parse your creative request. Try describing the visual mood you want. {e}"
        return {"messages": [AIMessage(content=err)], "final_response": err}
    
    # Call moodboard tool
    tools = await get_mcp_tools(servers=["moodboard"])
    moodboard_tool = next((t for t in tools if t.name == "generate_moodboard"), None)

    if not moodboard_tool:
        err = "⚠️ Moodboard tool not available. Is the moodboard MCP server running?"
        return {"messages": [AIMessage(content=err)], "final_response": err}
    
    try:
        raw = await moodboard_tool.ainvoke({
            "prompt": extracted.visual_prompt,
            "count": extracted.count,
        })
        result = json.loads(raw) if isinstance(raw, str) else raw
    except Exception as e:
        err = f"Image generation failed: {e}"
        return {"messages": [AIMessage(content=err)], "final_response": err}

    if result.get("error"):
        err = f"Image generation error: {result['error']}"
        return {"messages": [AIMessage(content=err)], "final_response": err}

    images = result.get("images", [])
    if not images:
        err = "No images were generated."
        return {"messages": [AIMessage(content=err)], "final_response": err}

    # Format response
    lines = [
        f"🎨 Here's your moodboard!\n",
        f"**Prompt used:** _{extracted.visual_prompt}_"
    ]

    for i, img in enumerate(images, 1):
        lines.append(f"**Image {i}:** {img['image_url']}")

    lines.append(
        "\n_Images are valid for 7 days. "
        "Let me know if you'd like a different style or mood!_"
    )

    response_text = "\n".join(lines)
    return {
        "messages": [AIMessage(content=response_text)],
        "final_response": response_text,
    }