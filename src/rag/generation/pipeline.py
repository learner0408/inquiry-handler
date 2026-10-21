"""Sole RAG orchestrator (sync). history.load -> cache.get -> HIT return /
MISS -> hybrid retrieval -> rerank -> answer -> cache.set + history.save.

Sync, matching every RAG leg; FastAPI webhooks will offload via
run_in_threadpool. Fallback answers are never cached (no citations = no
provenance), but ARE saved to history (the user asked; continuity matters).
"""

from dataclasses import dataclass, field

from src.config import get_settings
from src.conversation import history
from src.rag.cache.semantic_cache import SemanticCache
from src.rag.generation.answer import FALLBACK, generate_answer
from src.rag.reranking.reranker import NemotronReranker
from src.rag.retrieval.hybrid import HybridRetriever


@dataclass
class InquiryResult:
    text: str
    citations: list[str] = field(default_factory=list)
    cache_hit: bool = False
    fallback: bool = False


def _rerank_to_dicts(ranked: list) -> list[dict]:
    return [
        {"doc_id": c.doc_id, "revision_id": c.revision_id,
         "chunk_index": c.chunk_index, "text": c.text}
        for c in ranked
    ]


def handle_inquiry(
    channel: str, sender_id: str, text: str, external_id: str | None = None
) -> InquiryResult:
    history_turns = history.load(sender_id, channel)
    hit = SemanticCache().get(text)
    if hit is not None:
        return InquiryResult(text=hit.answer, citations=hit.citations, cache_hit=True)

    fused = HybridRetriever().retrieve(text, get_settings().TOP_K)
    reranked = _rerank_to_dicts(NemotronReranker().rerank(text, fused))
    if not reranked:
        history.save(sender_id, channel, text, FALLBACK, external_id=external_id)
        return InquiryResult(text=FALLBACK, fallback=True)

    generated = generate_answer(text, history_turns, reranked)
    answer = generated["answer"]
    if generated["fallback"]:
        history.save(sender_id, channel, text, answer, external_id=external_id)
        return InquiryResult(text=answer, fallback=True)

    top = reranked[0]
    SemanticCache().set(text, top["revision_id"], answer, generated["citations"], top["doc_id"])
    history.save(sender_id, channel, text, answer, external_id=external_id)
    return InquiryResult(text=answer, citations=generated["citations"])
