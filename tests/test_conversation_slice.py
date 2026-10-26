"""Conversation slice: summary context, titles, batched roll-ups, and retries."""

import socket
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import text as sa_text

from src.conversation import history, threads
from src.db.engine import get_sync_engine
from src.rag.generation.answer import (
    FALLBACK,
    generate_answer,
    generate_thread_title,
    summarize_conversation,
)


def _db_reachable() -> bool:
    try:
        with socket.create_connection(("localhost", 5432), timeout=2):
            return True
    except OSError:
        return False


def _sender() -> str:
    return "conversation-test-" + uuid.uuid4().hex[:8]


def _cleanup(sender: str) -> None:
    with get_sync_engine().begin() as conn:
        conn.execute(
            sa_text("DELETE FROM thread_messages WHERE thread_id IN "
                    "(SELECT id FROM conversation_threads WHERE sender_id = :s)"),
            {"s": sender},
        )
        conn.execute(
            sa_text("DELETE FROM conversation_threads WHERE sender_id = :s"),
            {"s": sender},
        )


def _thread_state(sender: str):  # type: ignore[no-untyped-def]
    with get_sync_engine().connect() as conn:
        return conn.execute(
            sa_text("SELECT title, summary, summary_timestamp FROM conversation_threads "
                    "WHERE sender_id = :s"),
            {"s": sender},
        ).one()


def test_generate_answer_keeps_summary_while_capping_raw_history() -> None:
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content=FALLBACK)
    turns = [
        {"role": "system", "content": "Earlier conversation summary: old facts"},
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "recent question"},
        {"role": "assistant", "content": "recent answer"},
    ]
    settings = SimpleNamespace(HISTORY_MAX_MESSAGES=2)
    with patch("src.rag.generation.answer.get_settings", return_value=settings), patch(
        "src.rag.generation.answer._get_llm", return_value=fake_llm
    ):
        generate_answer("new question", turns, [])
    contents = [message.content for message in fake_llm.invoke.call_args[0][0]]
    assert "Earlier conversation summary: old facts" in contents
    assert "recent question" in contents and "recent answer" in contents
    assert "old question" not in contents and "old answer" not in contents


def test_summary_and_title_prompts_receive_existing_context() -> None:
    fake_llm = MagicMock()
    fake_llm.invoke.side_effect = [MagicMock(content="merged summary"),
                                   MagicMock(content="Returns and refunds")]
    turns = [{"role": "user", "content": "Can I return it?"}]
    with patch("src.rag.generation.answer._get_llm", return_value=fake_llm):
        assert summarize_conversation("old summary", turns) == "merged summary"
        assert generate_thread_title(turns) == "Returns and refunds"
    summary_prompt = str(fake_llm.invoke.call_args_list[0].args[0][1].content)
    assert "old summary" in summary_prompt and "Can I return it?" in summary_prompt


@pytest.mark.skipif(not _db_reachable(), reason="no local postgres")
def test_history_load_prepends_stored_summary() -> None:
    sender = _sender()
    try:
        history.save(sender, "whatsapp", "Q1", "A1")
        with get_sync_engine().begin() as conn:
            conn.execute(
                sa_text("UPDATE conversation_threads SET summary = 'older facts' "
                        "WHERE sender_id = :s"),
                {"s": sender},
            )
        turns = history.load(sender, "whatsapp")
        assert turns[0] == {
            "role": "system",
            "content": "Earlier conversation summary: older facts",
        }
        assert [(turn["role"], turn["content"]) for turn in turns[1:]] == [
            ("user", "Q1"), ("assistant", "A1")
        ]
    finally:
        _cleanup(sender)


@pytest.mark.skipif(not _db_reachable(), reason="no local postgres")
def test_maintain_generates_title_only_once() -> None:
    sender = _sender()
    try:
        history.save(sender, "whatsapp", "What is the return window?", "Thirty days.")
        with patch("src.conversation.threads.generate_thread_title",
                   return_value="Return window") as title, patch(
            "src.conversation.threads.summarize_conversation"
        ) as summarize:
            threads.maintain("whatsapp", sender)
            threads.maintain("whatsapp", sender)
        assert title.call_count == 1
        assert summarize.call_count == 0
        assert _thread_state(sender).title == "Return window"
    finally:
        _cleanup(sender)


@pytest.mark.skipif(not _db_reachable(), reason="no local postgres")
def test_maintain_waits_for_full_delta_and_never_re_summarizes() -> None:
    sender = _sender()
    settings = SimpleNamespace(HISTORY_MAX_MESSAGES=3)
    try:
        with patch("src.conversation.threads.get_settings", return_value=settings), patch(
            "src.conversation.threads.generate_thread_title", return_value="Support thread"
        ), patch(
            "src.conversation.threads.summarize_conversation",
            side_effect=["summary one", "summary two"],
        ) as summarize:
            for number in range(1, 6):
                history.save(sender, "whatsapp", f"Q{number}", f"A{number}")
                threads.maintain("whatsapp", sender)
                if number == 4:
                    assert summarize.call_count == 1  # only 2 new expired rows: wait

        assert summarize.call_count == 2
        first_delta = summarize.call_args_list[0].args[1]
        second_delta = summarize.call_args_list[1].args[1]
        assert [turn["content"] for turn in first_delta] == ["Q1", "A1", "Q2"]
        assert [turn["content"] for turn in second_delta] == ["A2", "Q3", "A3", "Q4"]
        assert summarize.call_args_list[1].args[0] == "summary one"
        state = _thread_state(sender)
        assert state.summary == "summary two" and state.summary_timestamp is not None

        with patch("src.conversation.threads.summarize_conversation") as no_repeat:
            threads.maintain("whatsapp", sender)
        assert no_repeat.call_count == 0  # same data cannot trigger another merge
    finally:
        _cleanup(sender)
