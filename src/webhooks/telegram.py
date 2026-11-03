"""Telegram webhook: verify secret-token header, ack fast, dedupe, delegate.

POST flow: constant-time compare of X-Telegram-Bot-Api-Secret-Token vs
TELEGRAM_WEBHOOK_SECRET -> 403 on mismatch; else 200 {"ok": true} immediately,
then background task: parse -> non-text/edits ignored -> external_id dedupe
(check-first, unique backstop in history.save) -> handle_inquiry (sync
pipeline, offloaded via to_thread) -> send reply -> maintain thread
title/summary. Telegram only fires the webhook when the secret header
registered via setWebhook matches; no GET handshake exists.
"""

import asyncio
import hmac
import json
import logging

from fastapi import APIRouter, Request, Response
from sqlalchemy.exc import IntegrityError

from src.channels.telegram import TelegramMessage, parse_telegram_payload, send_message
from src.config import get_settings
from src.conversation.threads import maintain
from src.rag.generation.pipeline import handle_inquiry
from src.webhooks.idempotency import external_id_seen as _external_id_seen

logger = logging.getLogger(__name__)
router = APIRouter()

SECRET_HEADER = "X-Telegram-Bot-Api-Secret-Token"


@router.post("/webhooks/telegram")
async def inbound(request: Request) -> Response:
    raw_secret = request.headers.get(SECRET_HEADER, "")
    settings = get_settings()
    if not settings.TELEGRAM_WEBHOOK_SECRET:
        raise ValueError("TELEGRAM_WEBHOOK_SECRET must be set to verify webhooks")
    if not hmac.compare_digest(raw_secret, settings.TELEGRAM_WEBHOOK_SECRET):
        return Response(content='{"detail": "invalid secret token"}',
                        status_code=403, media_type="application/json")

    message = parse_telegram_payload(json.loads(await request.body()))
    if message is not None and not _external_id_seen(message.external_id):
        asyncio.create_task(_process(message))
    return Response(content=json.dumps({"ok": "true"}), media_type="application/json")


async def _process(message: TelegramMessage) -> None:
    try:
        result = await asyncio.to_thread(
            handle_inquiry, message.channel, message.sender_id, message.text,
            message.external_id,
        )
        await send_message(message.sender_id, result.text)
        await asyncio.to_thread(maintain, message.channel, message.sender_id)
    except IntegrityError:
        logger.info("dedupe backstop: external_id already processed")
    except Exception:  # noqa: BLE001  background task must never crash the loop
        logger.exception("telegram background processing failed")
