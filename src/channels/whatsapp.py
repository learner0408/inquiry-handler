"""WhatsApp adapter: parse Meta webhook payloads, send via Cloud API. No LLM here."""

import json
from dataclasses import dataclass

import httpx

from src.config import get_settings

GRAPH_BASE = "https://graph.facebook.com/v23.0"


@dataclass
class WhatsAppMessage:
    channel: str = "whatsapp"
    sender_id: str = ""
    text: str = ""
    external_id: str = ""


def parse_meta_payload(payload: dict) -> WhatsAppMessage | None:
    """Extract body.entry[0].changes[0].value.messages[0]; None for statuses."""
    try:
        value = payload["entry"][0]["changes"][0]["value"]
        msg = value["messages"][0]
        if msg.get("type") != "text":
            return None
        return WhatsAppMessage(
            sender_id=msg.get("from", ""),
            text=msg.get("text", {}).get("body", ""),
            external_id=msg.get("id", ""),
        )
    except (KeyError, IndexError):
        return None


def _payload(to: str, text: str) -> dict:
    return {"messaging_product": "whatsapp",
            "recipient_type": "individual", "to": to,
            "type": "text", "text": {"preview_url": False, "body": text}}


async def send_message(to: str, text: str) -> dict:
    """POST to Cloud API ground truth (text only). Raises on non-2xx Meta reply.

    Note: fresh portfolios may return 131031 until business verification clears;
    that is a Meta-side gate retried by the next inbound user message.
    """
    settings = get_settings()
    if not settings.WHATSAPP_ACCESS_TOKEN or not settings.WHATSAPP_PHONE_ID:
        raise ValueError("WHATSAPP_ACCESS_TOKEN and WHATSAPP_PHONE_ID must be set to send")
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{GRAPH_BASE}/{settings.WHATSAPP_PHONE_ID}/messages",
            headers={"Authorization": f"Bearer {settings.WHATSAPP_ACCESS_TOKEN}",
                     "Content-Type": "application/json"},
            json=_payload(to, text),
        )
        response.raise_for_status()
        return response.json()


def is_status_webhook(raw: bytes) -> bool:
    """True when the payload carries only statuses (delivery receipts), no message."""
    try:
        value = json.loads(raw)["entry"][0]["changes"][0]["value"]
    except (KeyError, IndexError, json.JSONDecodeError):
        return False
    return "messages" not in value
