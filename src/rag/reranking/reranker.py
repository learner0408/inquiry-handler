"""Cross-encoder rerank leg: NIM llama-nemotron-rerank-vl-1b-v2 via LangChain.

NVIDIARerank defaults to integrate.api.nvidia.com, whose tracked rerank models
are EOL (410 Gone as of 2026-05-18); the live hosted path embeds the model name
in the URL, so settings expose RERANK_BASE_URL explicitly. Input chunks are
tagged with their position (text can repeat across docs; identity is position,
not content). distance semantics flip here: rerank logit, higher = better.
"""

import threading

from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import NVIDIARerank

from src.config import get_settings
from src.rag.vectordb.chunks import ScoredChunk


class NemotronReranker:
    def __init__(self) -> None:
        settings = get_settings()
        self._client = NVIDIARerank(
            model=settings.RERANK_MODEL,
            base_url=settings.RERANK_BASE_URL,
            api_key=settings.NVIDIA_API_KEY,
            top_n=settings.RERANK_TOP_N,
        )
        self._lock = threading.Lock()

    def rerank(self, query: str, chunks: list[ScoredChunk]) -> list[ScoredChunk]:
        if not chunks:
            return []
        docs = [
            Document(page_content=c.text, metadata={"pos": i})
            for i, c in enumerate(chunks)
        ]
        with self._lock:  # stream client is not documented as thread-safe
            results = self._client.compress_documents(query=query, documents=docs)
        return [
            ScoredChunk(
                doc_id=chunks[d.metadata["pos"]].doc_id,
                revision_id=chunks[d.metadata["pos"]].revision_id,
                chunk_index=chunks[d.metadata["pos"]].chunk_index,
                text=d.page_content,
                distance=d.metadata["relevance_score"],
            )
            for d in results
        ]
