"""Sole LLM caller. RAG-only, low temp, chunk-level provenance, explicit fallback.

Citation contract (v1.1.0): the ANSWER text carries doc-level [doc_id] markers
only (WhatsApp users never see chunk indices); the structured citations list
carries doc_id#chunk_index provenance mapped from retrieval (top-1 reranked
chunk per cited doc), so downstream precision never depends on the model
emitting chunk indices. Hallucinated doc refs are dropped; zero valid refs ->
answer returned with empty citations (not a fallback).
"""

import re
from functools import lru_cache
from pathlib import Path

import yaml
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.config import get_settings

FALLBACK = "I couldn't find that in the available documents."
PROMPTS_PATH = Path("prompts/rag_prompts.yaml")


@lru_cache
def _load_prompts() -> dict:
    return yaml.safe_load(PROMPTS_PATH.read_text())


@lru_cache
def _get_llm() -> BaseChatModel:
    settings = get_settings()
    if settings.LLM_PROVIDER == "anthropic":
        if not settings.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY must be set when LLM_PROVIDER=anthropic")
        return ChatAnthropic(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            api_key=settings.ANTHROPIC_API_KEY,
        )
    if settings.LLM_PROVIDER == "openai":
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY must be set when LLM_PROVIDER=openai")
        from langchain_openai import ChatOpenAI  # why: lazy — anthropic is the default provider

        return ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            api_key=settings.OPENAI_API_KEY,
        )
    raise ValueError(f"unknown LLM_PROVIDER: {settings.LLM_PROVIDER}")


def build_context(context_chunks: list[dict]) -> str:
    """Chunk dicts keyed doc_id/text; labels doc-level only (no chunk indices)."""
    return "\n\n".join(f"[{c['doc_id']}]\n{c['text']}" for c in context_chunks)


def _parse_doc_refs(answer_text: str) -> list[str]:
    return re.findall(r"\[([0-9A-Za-z_-]{8,})\]", answer_text)


def resolve_citations(answer_text: str, reranked: list[dict]) -> list[str]:
    """Parse [doc_id] refs from the answer; map each cited doc -> top-1 chunk.

    reranked dicts: doc_id / revision_id / chunk_index / text. Returns
    doc_id#chunk_index provenance from retrieval (never model-emitted chunk
    indices); hallucinated refs dropped; no valid refs -> empty list.
    """
    top_chunk_per_doc: dict[str, dict] = {}
    for c in reranked:
        top_chunk_per_doc.setdefault(c["doc_id"], c)
    citations: list[str] = []
    for ref in _parse_doc_refs(answer_text):
        c = top_chunk_per_doc.get(ref)
        if c is not None:
            citation = f"{c['doc_id']}#{c['chunk_index']}"
            if citation not in citations:
                citations.append(citation)
    return citations


def generate_answer(query: str, history: list[dict], reranked: list[dict]) -> dict:
    settings = get_settings()
    prompt = _load_prompts()["prompts"]["rag_qa"]
    msgs = [SystemMessage(prompt["system"])]
    for turn in history[-settings.HISTORY_MAX_TURNS:]:
        if turn["role"] == "assistant":
            msgs.append(AIMessage(turn["content"]))
        else:
            msgs.append(HumanMessage(turn["content"]))
    ctx = build_context(reranked)
    msgs.append(HumanMessage(prompt["user_template"].format(ctx=ctx, query=query)))
    response = _get_llm().invoke(msgs)
    answer = (response.content or "").strip()
    return {"answer": answer, "citations": resolve_citations(answer, reranked),
            "fallback": answer == FALLBACK}
