"""Webhook idempotency: channel-neutral external_id seen-check."""

from sqlalchemy import text as sa_text

from src.db.engine import get_sync_engine

EXTERNAL_ID_SEEN = "SELECT 1 FROM thread_messages WHERE external_id = :mid"


def external_id_seen(external_id: str) -> bool:
    if not external_id:
        return False
    with get_sync_engine().connect() as conn:
        return conn.execute(sa_text(EXTERNAL_ID_SEEN), {"mid": external_id}).first() is not None
