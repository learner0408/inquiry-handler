"""Nemotron embeddings via LangChain NVIDIAEmbeddings (NIM cloud).

LangChain handles the passage/query contract: embed_passages -> input_type
passage, embed_query -> input_type query. truncate=END guards oversize input.
"""

from functools import lru_cache

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

from src.config import get_settings


@lru_cache
def get_embedder() -> NVIDIAEmbeddings:
    settings = get_settings()
    return NVIDIAEmbeddings(
        model=settings.EMBEDDING_MODEL,
        api_key=settings.NVIDIA_API_KEY,
        base_url=settings.NEMOTRON_EMBED_URL,
        truncate="END",
    )


def _assert_dims(vectors: list[list[float]]) -> None:
    expected = get_settings().VECTOR_DIMS
    for v in vectors:
        if len(v) != expected:
            raise ValueError(f"embedding dim {len(v)} != VECTOR_DIMS {expected}")


def embed_passages(texts: list[str]) -> list[list[float]]:
    vectors = get_embedder().embed_documents(texts)
    _assert_dims(vectors)
    return vectors


def embed_query(text: str) -> list[float]:
    vector = get_embedder().embed_query(text)
    _assert_dims([vector])
    return vector
