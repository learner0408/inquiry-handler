"""Pluggable chunking: Chunker protocol + registry. Default: recursive (LangChain).

Add a strategy by implementing Chunker and registering in CHUNKERS — callers
select via CHUNK_STRATEGY without code changes (sizes A/B-tested later).
"""

import hashlib
from dataclasses import dataclass
from typing import Callable, Protocol

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_settings


@dataclass
class Chunk:
    chunk_index: int
    text: str
    chunk_hash: str


def chunk_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


class Chunker(Protocol):
    def split(self, text: str) -> list[Chunk]: ...


class RecursiveChunker:
    """Recursive separator split; size/overlap from settings."""

    def split(self, text: str) -> list[Chunk]:
        settings = get_settings()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        return [
            Chunk(chunk_index=i, text=t, chunk_hash=chunk_hash(t))
            for i, t in enumerate(splitter.split_text(text))
        ]


CHUNKERS: dict[str, Callable[[], Chunker]] = {
    "recursive": RecursiveChunker,
}


def get_chunker() -> Chunker:
    strategy = get_settings().CHUNK_STRATEGY
    try:
        return CHUNKERS[strategy]()
    except KeyError:
        raise ValueError(f"unknown CHUNK_STRATEGY: {strategy} (known: {sorted(CHUNKERS)})") from None
