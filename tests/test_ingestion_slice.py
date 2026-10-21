"""Ingestion-slice tests: fixtures only, no live Google/NVIDIA calls."""

import socket
import uuid

import pytest

from src.rag.chunking.chunker import chunk_hash, get_chunker
from src.rag.embeddings import embedder as embedder_mod
from src.rag.ingestion.google_docs import fetch_doc


def _db_reachable() -> bool:
    try:
        with socket.create_connection(("localhost", 5432), timeout=2):
            return True
    except OSError:
        return False


def test_chunk_hash_deterministic_and_normalized() -> None:
    assert chunk_hash("hello") == chunk_hash("  hello\n")
    assert chunk_hash("hello") != chunk_hash("world")


def test_chunker_splits_and_orders() -> None:
    text = " ".join(f"Sentence number {i} carries distinct content here." for i in range(200))
    chunks = get_chunker().split(text)
    assert len(chunks) > 1
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert len({c.chunk_hash for c in chunks}) == len(chunks)


def test_chunker_unknown_strategy() -> None:
    import src.rag.chunking.chunker as chunker_mod
    from src.config import get_settings

    orig = get_settings().CHUNK_STRATEGY
    get_settings().CHUNK_STRATEGY = "nope"  # type: ignore[assignment]
    try:
        with pytest.raises(ValueError, match="unknown CHUNK_STRATEGY"):
            chunker_mod.get_chunker()
    finally:
        get_settings().CHUNK_STRATEGY = orig  # type: ignore[assignment]


def test_fetch_doc_rejects_non_allowlisted_without_network() -> None:
    with pytest.raises(ValueError, match="allowlist"):
        fetch_doc("definitely-not-allowlisted")


def test_embedder_dims_assert() -> None:
    embedder_mod._assert_dims([[0.0] * 2048])
    with pytest.raises(ValueError, match="VECTOR_DIMS"):
        embedder_mod._assert_dims([[0.0] * 3])


@pytest.mark.skipif(not _db_reachable(), reason="no local postgres")
def test_atomic_replace_orphan_delete() -> None:
    from sqlalchemy import text as sa_text

    from src.db.engine import get_sync_engine
    from src.rag.vectordb.chunks import ChunkRow, PgVectorChunks

    store = PgVectorChunks()
    doc = "test-doc-" + uuid.uuid4().hex[:8]
    vec = [0.01] * 2048
    v1 = [ChunkRow(0, "alpha beta", chunk_hash("alpha beta"), vec)]
    v2 = [ChunkRow(0, "gamma delta", chunk_hash("gamma delta"), vec)]
    try:
        s1 = store.atomic_replace(doc, "r1", v1)
        assert s1["inserted"] == 1
        s2 = store.atomic_replace(doc, "r2", v2)  # same doc, new revision
        assert s2["deleted"] == 1 and s2["inserted"] == 1
        s3 = store.atomic_replace(doc, "r2", v2)  # identical re-ingest: pure no-op
        assert s3["deleted"] == 0 and s3["inserted"] == 0
        hits = store.search(vec, top_k=5)
        assert any(h.doc_id == doc and h.revision_id == "r2" for h in hits)
    finally:
        engine = get_sync_engine()
        with engine.begin() as conn:
            conn.execute(sa_text("DELETE FROM chunks WHERE doc_id = :d"), {"d": doc})
