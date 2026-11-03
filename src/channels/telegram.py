"""Telegram adapter: parse Bot API updates, send via sendMessage. No LLM here."""

from dataclasses import dataclass

import httpx

from src.config import get_settings

TELEGRAM_API_BASE = "https://api.telegram.org"


@dataclass
class TelegramMessage:
    channel: str = "telegram"
    sender_id: str = ""
    text: str = ""
    external_id: str = ""


def parse_telegram_payload(payload: dict) -> TelegramMessage | None:
    """Extract update.message text; None for edits, callbacks, and non-text.

    sender_id is the CHAT id (the reply target); external_id is the
    update_id, which is globally unique per bot — dedupe key in history.
    """
    msg = payload.get("message")
    if not isinstance(msg, dict) or "text" not in msg:
        return None
    return TelegramMessage(
        sender_id=str(msg.get("chat", {}).get("id", "")),
        text=str(msg.get("text", "")),
        external_id=str(payload.get("update_id", "")),
    )


async def send_message(to: str, text: str) -> dict:
    """POST sendMessage ground truth (plain text; no parse_mode). Raises on
    non-2xx — Telegram caps sendMessage at 4096 chars."""
    settings = get_settings()
    if not settings.TELEGRAM_BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN must be set to send")
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{TELEGRAM_API_BASE}/bot{settings.TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": to, "text": text},
        )
        response.raise_for_status()
        return response.json()
