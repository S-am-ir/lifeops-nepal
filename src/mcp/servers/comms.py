import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel
from typing import Optional
from src.config.settings import settings
import httpx

mcp = FastMCP("comms", json_response=True)

class WhatsAppResult(BaseModel):
    status: str
    message_id: Optional[str] = None
    error: Optional[str] = None

@mcp.tool()
async def send_whatsapp_message(to_number: str, body: str) -> WhatsAppResult:
    """Send an outbound WhatsApp message via the WhatsApp Cloud API.
    
    Use this to deliver reminder, summaries, alerts, or any
    notification to the user's WhatsApp. This is outbound-only - it sends 
    a message it does not receive or process replies.
    
    Call this tool when:
    - The user asks to be reminded about something (flight time, hotel check-in , whether alert)
    - You have built a travel summary or itinerary and the user wants it sent to WhatsApp
    - A scheduled or triggered notification needs to be delivered
    
    Args:
        to_number: Recipient phone number in international format without the +.
                   For Nepal numbers: 977 followed by the 10-digit number.
                   E.g. "9779812345678" for +977 981-234-5678
        body:      Message text to send. Plain text only, max 4096 characters.
                   Be concise — this is a WhatsApp message, not an email.

    Returns:
        WhatsAppResult with status "sent" and message_id on success,
        or status "error" and an error description on failure.
    
    Example:
        send_whatsapp_message(
            to_number="9779812345678",
            body="Reminder: Your flight KTM→PKR departs tomorrow at 07:30. Check-in opens at 05:30."
        )
    """ 
    url = f"https://graph.facebook.com/v21.0/{settings.whatsapp_phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {settings.whatsapp_access_token.get_secret_value()}",
        "Content-Type": "application/json",
    }

    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to_number,
        "type": "text",
        "text": {"body": body},
    }

    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return WhatsAppResult(
                status="sent",
                message_id=data["messages"][0]["id"],
            )
        except httpx.HTTPStatusError as e:
            return WhatsAppResult(
                status="error",
                error=f"HTTP {e.response.status_code}: {e.response.text}"         
            )
        except Exception as e:
            return WhatsAppResult(
                status="error",
                error=str(e),
            )
        
if __name__ == "__main__":
    print(f"[MCP Comms] running on port {settings.mcp_comms_port}")
    mcp.run(transport="streamable-http")