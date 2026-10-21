"""Retrieval-slice tests: RRF fusion (pure) + lexical/hybrid vs real DB.

Dense leg swaps the embedder for a deterministic fake so hybrid tests cover the
full fused path offline; reranker is tested behind a stubbed NVIDIARerank.
"""

import socket
import uuid
from unittest.mock import MagicMock, patch

import pytest

from src.rag.retrieval.bm25 import LexicalRetriever
from src.rag.retrieval.hybrid import HybridRetriever, rrf_fuse
from src.rag.vectordb.chunks import ChunkRow, ScoredChunk


def _db_reachable() -> bool:
    try:
        with socket.create_connection(("localhost", 5432), timeout=2):
            return True
    except OSError:
        return False


def _chunk(doc: str, idx: int, text: str) -> ScoredChunk:
    return ScoredChunk(doc_id=doc, revision_id="r1", chunk_index=idx, text=text, distance=0.0)


def _keys(chunks: list[ScoredChunk]) -> list[int]:
    return [c.chunk_index for c in chunks]


def test_rrf_fuse_prefers_both_list_hits() -> None:
    a = _chunk("d", 0, "both hit")
    b = _chunk("d", 1, "dense only")
    c = _chunk("d", 2, "lexical only")
    fused = rrf_fuse([[a, b], [a, c]])
    assert _keys(fused) == [0, 1, 2]  # both-list chunk first, then stable tie order


def test_rrf_fuse_deterministic_on_ties() -> None:
    a = _chunk("d", 0, "a")
    b = _chunk("d", 1, "b")
    assert _keys(rrf_fuse([[a], [b]])) == [0, 1]  # equal scores -> first-seen order


def test_rrf_fuse_rank_order() -> None:
    # 1/(k + rank) keeps intra-list rank order: rank-1 beats rank-2
    a = _chunk("d", 10, "rank1")
    b = _chunk("d", 11, "rank2")
    assert _keys(rrf_fuse([[a, b]])) == [10, 11]


def test_lexical_retriever_empty_query_is_safe() -> None:
    if not _db_reachable():
        pytest.skip("no local postgres")
    # websearch_to_tsquery("") matches nothing; must not raise
    assert LexicalRetriever().retrieve("", top_k=3) == []


def test_hybrid_rescues_lexical_only_hits() -> None:
    """Keyword match on a near-zero-vector chunk: dense misses, fusion keeps it."""
    if not _db_reachable():
        pytest.skip("no local postgres")

    from sqlalchemy import text as sa_text

    from src.db.engine import get_sync_engine
    from src.rag.vectordb.chunks import PgVectorChunks

    doc = "test-doc-" + uuid.uuid4().hex[:8]
    store = PgVectorChunks()
    tiny = [0.001] * 2048  # near-zero for every real embedding
    keyword_text = "ZKXQW: ordnance part number 88-441"
    store.atomic_replace(
        doc,
        "r1",
        [
            ChunkRow(0, "alpha beta gamma", uuid.uuid4().hex, [1.0] * 2048),
            ChunkRow(1, keyword_text, uuid.uuid4().hex, tiny),
        ],
    )
    try:
        with patch(
            "src.rag.retrieval.hybrid.embed_query",
            MagicMock(return_value=[1.0] * 2048),
        ):
            fused = HybridRetriever().retrieve("ordnance part number 88-441", top_k=5)
        assert any(h.text == keyword_text for h in fused), "BM25-only hit lost by fusion"
    finally:
        with get_sync_engine().begin() as conn:
            conn.execute(sa_text("DELETE FROM chunks WHERE doc_id = :d"), {"d": doc})


def test_reranker_stubbed_mapping_and_order() -> None:
    """No network: stub compress_documents, assert pos->chunk mapping + order."""
    from src.rag.reranking.reranker import NemotronReranker

    chunks = [
        ScoredChunk(doc_id="d", revision_id="r", chunk_index=0, text="alpha", distance=0.9),
        ScoredChunk(doc_id="d", revision_id="r", chunk_index=1, text="beta", distance=0.8),
        ScoredChunk(doc_id="d", revision_id="r", chunk_index=2, text="gamma", distance=0.7),
    ]
    fake = MagicMock()

    def fake_compress(query, documents):  # noqa: ARG001
        tagged = {d.metadata["pos"]: d for d in documents}
        # server returns [gamma, alpha] best-first, by input-document identity
        return [
            tagged[2].model_copy(
                update={"metadata": {**tagged[2].metadata, "relevance_score": -1.2}}
            ),
            tagged[0].model_copy(
                update={"metadata": {**tagged[0].metadata, "relevance_score": -2.0}}
            ),
        ]

    fake.compress_documents.side_effect = fake_compress
    with patch("src.rag.reranking.reranker.NVIDIARerank", MagicMock(return_value=fake)):
        out = NemotronReranker().rerank("q", chunks)
    assert [(c.chunk_index, c.text, c.distance) for c in out] == [
        (2, "gamma", -1.2),
        (0, "alpha", -2.0),
    ]
