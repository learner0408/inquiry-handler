"""Conversation history I/O: thread lookup, summary + last-N read, paired save.

`load` returns an optional system summary followed by the most recent raw
messages. `save` records user and assistant turns together; the user row
carries external_id for webhook dedupe. Summaries and titles are maintained
by `threads.py` after replies are sent.
"""

from sqlalchemy import text as sa_text

from src.config import get_settings
from src.db.engine import get_sync_engine


def load(sender_id: str, channel: str) -> list[dict]:
    settings = get_settings()
    engine = get_sync_engine()
    with engine.connect() as conn:
        thread = conn.execute(
            sa_text(
                "SELECT id, summary FROM conversation_threads "
                "WHERE channel = :c AND sender_id = :s"
            ),
            {"c": channel, "s": sender_id},
        ).one_or_none()
        if thread is None:
            return []
        thread_id, summary = thread
        rows = conn.execute(
            sa_text(
                "SELECT role, content FROM thread_messages "
                "WHERE thread_id = :t ORDER BY created_at DESC LIMIT :n"
            ),
            {"t": str(thread_id), "n": settings.HISTORY_MAX_MESSAGES},
        ).all()
    turns = [{"role": r.role, "content": r.content} for r in reversed(rows)]
    if summary:
        turns.insert(0, {
            "role": "system",
            "content": f"Earlier conversation summary: {summary}",
        })
    return turns


def save(
    sender_id: str,
    channel: str,
    user_text: str,
    assistant_text: str,
    external_id: str | None = None,
) -> None:
    """Insert the user turn (carrying Meta message.id for dedupe when given) + assistant turn.

    An IntegrityError on the unique external_id means a duplicate/retried
    webhook raced through the check-first gate; callers treat it as "already
    answered" and skip the reply. Whole-transaction rollback on conflict is
    intended there (both turns of the dupe vanish).
    """
    engine = get_sync_engine()
    with engine.begin() as conn:
        thread_id = conn.execute(
            sa_text(
                "SELECT id FROM conversation_threads WHERE channel = :c AND sender_id = :s"
            ),
            {"c": channel, "s": sender_id},
        ).scalar_one_or_none()
        if thread_id is None:
            thread_id = conn.execute(
                sa_text(
                    "INSERT INTO conversation_threads (id, channel, sender_id, "
                    "created_at, updated_at) "
                    "VALUES (gen_random_uuid(), :c, :s, now(), now()) RETURNING id"
                ),
                {"c": channel, "s": sender_id},
            ).scalar_one()
        else:
            conn.execute(
                sa_text(
                    "UPDATE conversation_threads SET updated_at = now() WHERE id = :t"
                ),
                {"t": str(thread_id)},
            )
        conn.execute(
            sa_text(
                "INSERT INTO thread_messages (id, thread_id, role, content, created_at, external_id) "
                "VALUES (gen_random_uuid(), :t, 'user', :u, now(), :eid), "
                "(gen_random_uuid(), :t, 'assistant', :a, now() + interval '1 microsecond', NULL)"
            ),
            {"t": str(thread_id), "u": user_text, "a": assistant_text, "eid": external_id},
        )
