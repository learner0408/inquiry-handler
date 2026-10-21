"""Semantic-cache tests: fixtures + mocked embedder; DB-backed when reachable."""

import socket
import uuid
from unittest.mock import patch

import pytest

from src.rag.cache.semantic_cache import CacheHit, SemanticCache, query_hash
from src.rag.vectordb.chunks import ChunkRow


def _db_reachable() -> bool:
    try:
        with socket.create_connection(("localhost", 5432), timeout=2):
            return True
    except OSError:
        return False


def test_query_hash_normalizes_whitespace_and_case() -> None:
    assert query_hash("What is the Return Policy?") == query_hash("  what is the return policy? ")
    assert query_hash("a") != query_hash("b")


def test_set_refuses_blank_answer() -> None:
    with pytest.raises(ValueError, match="blank answer"):
        SemanticCache().set("q", "r1", "  ", ["c"], "d")


def _with_fake_embedder(vectors: list[list[float]]):
    calls = iter(vectors)
    return patch(
        "src.rag.cache.semantic_cache.embed_query",
        side_effect=lambda _text: next(calls),
    )


def _seed_chunk(doc: str, revision_id: str) -> None:
    """Cache freshness derives from chunks (one current revision per doc); tests
    must mirror that invariant or every entry reads as stale."""
    from src.rag.vectordb.chunks import PgVectorChunks

    PgVectorChunks().atomic_replace(
        doc,
        revision_id,
        [ChunkRow(0, f"seed text {doc}", uuid.uuid4().hex, [0.5] * 2048)],
    )


def _cleanup(doc: str) -> None:  # type: ignore[no-untyped-def]
    from sqlalchemy import text as sa_text

    from src.db.engine import get_sync_engine

    with get_sync_engine().begin() as conn:
        conn.execute(sa_text("DELETE FROM cache_entries WHERE doc_id = :d"), {"d": doc})
        conn.execute(sa_text("DELETE FROM chunks WHERE doc_id = :d"), {"d": doc})


@pytest.mark.skipif(not _db_reachable(), reason="no local postgres")
def test_cache_roundtrip_and_stale_revision() -> None:
    doc = "cache-test-" + uuid.uuid4().hex[:8]
    cache = SemanticCache()
    _seed_chunk(doc, "rev-A")
    try:
        with _with_fake_embedder([[1.0] * 2048, [1.0] * 2048]):
            cache.set("what is the return policy?", "rev-A",
                      "Returns within 30 days.", ["d/12"], doc)
        hit = cache.get("What is the return policy?")  # exact via Tier-1
        assert hit is not None
        assert isinstance(hit, CacheHit)
        assert (hit.answer, hit.citations, hit.revision_id) == (
            "Returns within 30 days.", ["d/12"], "rev-A"
        )
        assert hit.query_text == "what is the return policy?"  # stored original

        # semantic near-miss within threshold via Tier-2 (new question text)
        with _with_fake_embedder([[1.0] * 2048]):
            hit2 = cache.get("tell me the return policy please")
        assert hit2 is not None and hit2.answer == "Returns within 30 days."

        # revision bump: same question hash, new revision: old entry pruned,
        # fresh one replaces it in the same transaction
        _seed_chunk(doc, "rev-B")  # atomic replace: chunks now on rev-B
        with _with_fake_embedder([[1.0] * 2048, [1.0] * 2048]):
            cache.set("what is the return policy?", "rev-B",
                      "Returns within 14 days.", ["d/13"], doc)
        from sqlalchemy import text as sa_text

        from src.db.engine import get_sync_engine

        with get_sync_engine().connect() as conn:
            revs = conn.execute(
                sa_text("SELECT DISTINCT revision_id FROM cache_entries WHERE doc_id = :d"),
                {"d": doc},
            ).all()
            assert revs == [("rev-B",)]  # rev-A invalidated on bump
        refreshed = cache.get("what is the return policy?")
        assert refreshed is not None and refreshed.answer == "Returns within 14 days."

        # far-away embedding -> miss
        with _with_fake_embedder([[0.0] * 2048]):
            miss = cache.get("completely unrelated question about quantum physics")
        assert miss is None
    finally:
        _cleanup(doc)


@pytest.mark.skipif(not _db_reachable(), reason="no local postgres")
def test_hit_count_bumps_on_repeated_get() -> None:
    from sqlalchemy import text as sa_text

    from src.db.engine import get_sync_engine

    doc = "cache-test-" + uuid.uuid4().hex[:8]
    cache = SemanticCache()
    _seed_chunk(doc, "rev-A")
    try:
        with _with_fake_embedder([[1.0] * 2048] * 4):
            cache.set("shipping cost question", "rev-A", "5-7 business days.", [], doc)
            assert cache.get("shipping cost question") is not None  # Tier-1, bump 1
            assert cache.get("shipping cost problem") is not None   # Tier-2, bump 2
        with get_sync_engine().connect() as conn:
            n = conn.execute(
                sa_text("SELECT hit_count FROM cache_entries WHERE doc_id = :d"),
                {"d": doc},
            ).scalar_one()
        assert n == 2
    finally:
        _cleanup(doc)
