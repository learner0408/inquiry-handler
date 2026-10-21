"""Generation-slice tests: pipeline orchestration + history, fully stubbed legs.

No live LLM/reranker/retriever calls; DB-backed history roundtrip when
postgres is reachable. No LLM key required anywhere — answer/_get_llm is
always patched in these tests.
"""

import socket
import uuid
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from src.rag.generation.answer import FALLBACK, build_context, generate_answer, resolve_citations
from src.rag.generation.pipeline import InquiryResult, handle_inquiry
from src.rag.vectordb.chunks import ScoredChunk


def _db_reachable() -> bool:
    try:
        with socket.create_connection(("localhost", 5432), timeout=2):
            return True
    except OSError:
        return False


def _reranked() -> list[dict]:
    return [
        {"doc_id": "docA1234", "revision_id": "r1", "chunk_index": 7, "text": "alpha"},
        {"doc_id": "docB5678", "revision_id": "r1", "chunk_index": 2, "text": "beta"},
    ]


def _scored(doc: str, idx: int) -> ScoredChunk:
    return ScoredChunk(doc_id=doc, revision_id="r1", chunk_index=idx, text="t", distance=0.5)


# ---------- answer.py unit behavior (no LLM client) ----------


def test_resolve_citations_top1_and_hallucination_drop() -> None:
    text = "Shipping varies [docA1234]; customs duties [docB5678]; and [nope9999] is fake."
    assert resolve_citations(text, _reranked()) == ["docA1234#7", "docB5678#2"]
    assert resolve_citations("no citations offered", _reranked()) == []


def test_resolve_citations_dedupes_and_prefers_best_rank() -> None:
    reranked = [
        {"doc_id": "docA1234", "revision_id": "r1", "chunk_index": 9, "text": "best"},
        {"doc_id": "docA1234", "revision_id": "r1", "chunk_index": 3, "text": "later"},
    ]
    out = resolve_citations("twice [docA1234] and [docA1234]", reranked)
    assert out == ["docA1234#9"]  # first (best-rank) chunk wins, dedupe applied


def test_build_context_doc_level_labels_only() -> None:
    ctx = build_context(_reranked())
    assert "[docA1234]" in ctx and "#7" not in ctx  # chunk index never in context


def test_generate_answer_grounded_and_fallback_flags() -> None:
    fake_llm = MagicMock()

    with patch("src.rag.generation.answer._get_llm", return_value=fake_llm):
        fake_llm.invoke.return_value = MagicMock(content=FALLBACK)
        out = generate_answer("q", [], _reranked())
        assert out == {"answer": FALLBACK, "citations": [], "fallback": True}

        fake_llm.invoke.return_value = MagicMock(
            content="International shipping depends on weight [docA1234]."
        )
        out2 = generate_answer("q", [{"role": "user", "content": "prior"}], _reranked())
        assert out2["citations"] == ["docA1234#7"] and out2["fallback"] is False
        msgs = fake_llm.invoke.call_args[0][0]
        assert any("prior" in str(m.content) for m in msgs)  # history reached prompt


# ---------- history (DB-backed) ----------


def _cleanup_history(sender: str) -> None:
    from sqlalchemy import text as sa_text

    from src.db.engine import get_sync_engine

    with get_sync_engine().begin() as conn:
        conn.execute(
            sa_text("DELETE FROM thread_messages WHERE thread_id IN "
                    "(SELECT id FROM conversation_threads WHERE sender_id = :s)"),
            {"s": sender},
        )
        conn.execute(
            sa_text("DELETE FROM conversation_threads WHERE sender_id = :s"), {"s": sender}
        )


@pytest.mark.skipif(not _db_reachable(), reason="no local postgres")
def test_history_empty_then_save_then_order() -> None:
    from src.conversation import history

    sender = "hist-" + uuid.uuid4().hex[:8]
    try:
        assert history.load(sender, "whatsapp") == []  # no thread yet
        history.save(sender, "whatsapp", "Q1", "A1")
        history.save(sender, "whatsapp", "Q2", "A2")
        turns = history.load(sender, "whatsapp")
        assert [t["role"] for t in turns] == ["user", "assistant", "user", "assistant"]
        assert turns[-2]["content"] == "Q2" and turns[-1]["content"] == "A2"
    finally:
        _cleanup_history(sender)


# ---------- pipeline orchestration (all legs stubbed) ----------


@contextmanager
def _stubs(cache_get, cache_set, retrieves, reranks, generates):
    cache_cls = MagicMock(return_value=MagicMock(get=cache_get, set=cache_set))
    with patch("src.rag.generation.pipeline.SemanticCache", cache_cls), patch(
        "src.rag.generation.pipeline.HybridRetriever"
    ) as hyb, patch(
        "src.rag.generation.pipeline.NemotronReranker"
    ) as rer, patch(
        "src.rag.generation.pipeline.generate_answer", side_effect=generates
    ), patch(
        "src.rag.generation.pipeline.history"
    ) as hist:
        hyb.return_value = MagicMock(retrieve=lambda q, k: retrieves(q, k))
        rer.return_value = MagicMock(rerank=lambda q, ch: reranks(q, ch))
        yield hist


def test_pipeline_cache_hit_short_circuits() -> None:
    hit = MagicMock(answer="cached answer", citations=["d#1"], revision_id="r")
    gen_calls = []
    with _stubs(
        cache_get=lambda q: hit,
        cache_set=lambda *a: None,
        retrieves=lambda q, k: [_scored("d", 0)],
        reranks=lambda q, ch: {},
        generates=lambda q, h, r: gen_calls.append((q, h, r)) or {},
    ) as hist:
        res = handle_inquiry("whatsapp", "u1", "question")
    assert res == InquiryResult(text="cached answer", citations=["d#1"], cache_hit=True)
    assert gen_calls == [] and hist.save.call_count == 0  # hit: no LLM, no save


def test_pipeline_miss_grounded_caches_and_saves() -> None:
    cache_set_calls = []
    with _stubs(
        cache_get=lambda q: None,
        cache_set=lambda q, rev, ans, cit, doc: cache_set_calls.append((doc, rev, cit)),
        retrieves=lambda q, k: [_scored("docA1234", 7)],
        reranks=lambda q, ch: [_scored("docA1234", 7)],
        generates=lambda q, h, r: {
            "answer": "Grounded [docA1234].", "citations": ["docA1234#7"],
            "fallback": False},
    ) as hist:
        res = handle_inquiry("whatsapp", "u2", "question")
    assert res.text == "Grounded [docA1234]."
    assert res.citations == ["docA1234#7"] and res.cache_hit is False
    assert cache_set_calls == [("docA1234", "r1", ["docA1234#7"])]
    assert hist.save.call_count == 1 and hist.load.call_count == 1


def test_pipeline_fallback_never_cached_but_saved() -> None:
    cache_set_calls = []
    with _stubs(
        cache_get=lambda q: None,
        cache_set=lambda *a: cache_set_calls.append(a),
        retrieves=lambda q, k: [_scored("d", 0)],
        reranks=lambda q, ch: [_scored("d", 0)],
        generates=lambda q, h, r: {"answer": FALLBACK, "citations": [], "fallback": True},
    ) as hist:
        res = handle_inquiry("whatsapp", "u3", "question")
    assert res.text == FALLBACK and res.fallback is True
    assert cache_set_calls == []  # load-bearing: fallbacks never cached
    assert hist.save.call_count == 1  # but the turn is saved


def test_pipeline_empty_retrieval_fallback_skips_llm() -> None:
    gen_calls = []
    with _stubs(
        cache_get=lambda q: None,
        cache_set=lambda *a: None,
        retrieves=lambda q, k: [],
        reranks=lambda q, ch: [_scored("d", 0)] if ch else [],
        generates=lambda q, h, r: gen_calls.append((q, h, r)) or {},
    ):
        res = handle_inquiry("whatsapp", "u4", "question")
    assert res.text == FALLBACK and res.fallback is True
    assert gen_calls == []  # empty fused retrieval -> fallback before the LLM
