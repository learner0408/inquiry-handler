"""Hybrid retrieval: dense (pgvector) + lexical (FTS) fused by RRF.

RRF is scale-free (fuses RANKS, not scores): score = sum(1/(k + rank)) over the
lists a chunk appears in. Pure `rrf_fuse` is unit-testable without a DB.
"""

from src.rag.embeddings.embedder import embed_query
from src.rag.retrieval.bm25 import LexicalRetriever
from src.rag.vectordb.chunks import PgVectorChunks, ScoredChunk

RRF_K = 60


def _key(c: ScoredChunk) -> str:
    return f"{c.doc_id}:{c.chunk_index}"


def rrf_fuse(rankings: list[list[ScoredChunk]], k: int = RRF_K) -> list[ScoredChunk]:
    """Fuse ranked lists by reciprocal rank; preserve best list order on ties."""
    best: dict[str, ScoredChunk] = {}
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking):
            key = _key(chunk)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            if key not in best:
                best[key] = chunk
    # Re-sort with only required ordering key; stable sort keeps list order for ties
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    return [best[key] for key, _ in ranked]


class HybridRetriever:
    def __init__(self) -> None:
        self._dense = PgVectorChunks()
        self._lexical = LexicalRetriever()

    def retrieve(self, query: str, top_k: int) -> list[ScoredChunk]:
        dense = self._dense.search(embed_query(query), top_k)
        lexical = self._lexical.retrieve(query, top_k)
        return rrf_fuse([dense, lexical])[:top_k]
