"""Thread lifecycle: one-time titles and batched running summaries."""

from sqlalchemy import text as sa_text

from src.config import get_settings
from src.db.engine import get_sync_engine
from src.rag.generation.answer import generate_thread_title, summarize_conversation


def maintain(channel: str, sender_id: str) -> None:
    """Fill a missing title, then fold a full expired window into the summary.

    The answer path is untouched: this runs after the reply is sent. A failed
    LLM call leaves the corresponding marker unchanged so a later call retries.
    """
    engine = get_sync_engine()
    with engine.connect() as conn:
        thread = conn.execute(
            sa_text(
                "SELECT id, title, summary, summary_timestamp FROM conversation_threads "
                "WHERE channel = :c AND sender_id = :s"
            ),
            {"c": channel, "s": sender_id},
        ).one_or_none()
        if thread is None:
            return
        if not thread.title:
            first_turns = conn.execute(
                sa_text(
                    "SELECT role, content FROM thread_messages "
                    "WHERE thread_id = :t ORDER BY created_at LIMIT 2"
                ),
                {"t": str(thread.id)},
            ).all()
        else:
            first_turns = []

    if not thread.title and first_turns:
        title = generate_thread_title(
            [{"role": row.role, "content": row.content} for row in first_turns]
        )
        with engine.begin() as conn:
            conn.execute(
                sa_text("UPDATE conversation_threads SET title = :title "
                        "WHERE id = :t AND title = ''"),
                {"title": title, "t": str(thread.id)},
            )

    settings = get_settings()
    with engine.connect() as conn:
        message_count = conn.execute(
            sa_text("SELECT count(*) FROM thread_messages WHERE thread_id = :t"),
            {"t": str(thread.id)},
        ).scalar_one()
        if message_count < 2 * settings.HISTORY_MAX_MESSAGES:
            return
        window_start = conn.execute(
            sa_text(
                "SELECT created_at FROM thread_messages WHERE thread_id = :t "
                "ORDER BY created_at DESC OFFSET :offset LIMIT 1"
            ),
            {"t": str(thread.id), "offset": settings.HISTORY_MAX_MESSAGES - 1},
        ).scalar_one()
        delta_filter = " AND created_at > :summary_timestamp" if thread.summary_timestamp else ""
        delta = conn.execute(
            sa_text(
                "SELECT role, content, created_at FROM thread_messages "
                "WHERE thread_id = :t AND created_at < :window_start"
                + delta_filter + " ORDER BY created_at"
            ),
            {
                "t": str(thread.id),
                "window_start": window_start,
                "summary_timestamp": thread.summary_timestamp,
            },
        ).all()

    if not delta:
        return
    if len(delta) < settings.HISTORY_MAX_MESSAGES:
        return  # wait for a full expired batch; never summarize on every save
    delta_turns = [{"role": row.role, "content": row.content} for row in delta]
    summary = summarize_conversation(thread.summary, delta_turns)
    with engine.begin() as conn:
        conn.execute(
            sa_text(
                "UPDATE conversation_threads SET summary = :summary, "
                "summary_timestamp = :summary_timestamp WHERE id = :t"
            ),
            {
                "summary": summary,
                "summary_timestamp": delta[-1].created_at,
                "t": str(thread.id),
            },
        )
