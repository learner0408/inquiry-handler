"""WhatsApp webhook: verify signature, ack fast, dedupe, delegate to pipeline.

POST flow: raw-body HMAC-SHA256 vs X-Hub-Signature-256 (WHATSAPP_APP_SECRET) ->
403 on mismatch; else 200 {"ok": true} immediately, then background task:
parse -> statuses ignored -> external_id dedupe (check-first, unique backstop)
-> handle_inquiry (sync pipeline, offloaded via to_thread) -> send reply.
App secret enforced lazily here; test-WABA lock (error 131031) on fresh
portfolios is a Meta-side business-verification gate, not a code concern.
"""

import asyncio
import hashlib
import hmac
import json
import logging

from fastapi import APIRouter, Request, Response
from sqlalchemy import text as sa_text
from sqlalchemy.exc import IntegrityError

from src.channels.whatsapp import WhatsAppMessage, parse_meta_payload, send_message
from src.config import get_settings
from src.db.engine import get_sync_engine
from src.rag.generation.pipeline import handle_inquiry

logger = logging.getLogger(__name__)
router = APIRouter()

EXTERNAL_ID_SEEN = "SELECT 1 FROM thread_messages WHERE external_id = :mid"


def _bad_signature() -> Response:
    return Response(content='{"detail": "invalid signature"}',
                    status_code=403, media_type="application/json")


@router.get("/webhooks/whatsapp")
def verify(request: Request) -> Response:
    # Meta sends DOT-keyed query params: hub.mode, hub.verify_token, hub.challenge
    params = request.query_params
    settings = get_settings()
    if (params.get("hub.mode") == "subscribe"
            and params.get("hub.verify_token") == settings.WHATSAPP_VERIFY_TOKEN):
        challenge = params.get("hub.challenge", "")
        return Response(content=challenge, media_type="text/plain")
    return Response(content="forbidden", status_code=403, media_type="text/plain")


@router.post("/webhooks/whatsapp")
async def inbound(request: Request) -> dict[str, str]:
    raw = await request.body()
    settings = get_settings()
    if not settings.WHATSAPP_APP_SECRET:
        raise ValueError("WHATSAPP_APP_SECRET must be set to verify webhook signatures")
    supplied = request.headers.get("X-Hub-Signature-256", "")
    expected = "sha256=" + hmac.new(settings.WHATSAPP_APP_SECRET.encode(), raw, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(supplied, expected):
        return _bad_signature()

    message = parse_meta_payload(json.loads(raw))
    if message is not None and not _external_id_seen(message.external_id):
        asyncio.create_task(_process(message))
    return {"ok": "true"}


def _external_id_seen(external_id: str) -> bool:
    if not external_id:
        return False
    with get_sync_engine().connect() as conn:
        return conn.execute(sa_text(EXTERNAL_ID_SEEN), {"mid": external_id}).first() is not None


async def _process(message: WhatsAppMessage) -> None:
    try:
        result = await asyncio.to_thread(
            handle_inquiry, message.channel, message.sender_id, message.text,
            message.external_id,
        )
        await send_message(message.sender_id, result.text)
    except IntegrityError:
        logger.info("dedupe backstop: external_id already processed")
    except Exception:  # noqa: BLE001  background task must never crash the loop
        logger.exception("whatsapp background processing failed")
