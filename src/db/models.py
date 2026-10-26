"""Database tables (create_all for fresh DBs; Alembic for later changes).

chunks: RAG index rows (Nemotron passage embeddings, 2048 dims).
conversation_threads/thread_messages: durable multi-turn history.
cache_entries: 2-tier semantic cache (query_hash exact -> HNSW semantic).
"""

import uuid
from datetime import UTC, datetime

from sqlalchemy import (
    ARRAY,
    CheckConstraint,
    Computed,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Text,
    UniqueConstraint,
)
from sqlalchemy import (
    text as sa_text,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import UserDefinedType

from src.config import get_settings
from src.db.engine import Base


class HalfVec(UserDefinedType):
    """halfvec(D) column type (pgvector>=0.3 client lacks it; server 0.8.x supports it).

    HNSW caps vector() at 2000 dims; halfvec allows 4000 — required for 2048-dim
    Nemotron embeddings. See ARCHITECTURE.md.
    """

    cache_ok = True

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def get_col_spec(self, **kw: object) -> str:
        return f"HALFVEC({self.dim})"

    def bind_processor(self, dialect):  # type: ignore[no-untyped-def]
        def process(value):  # type: ignore[no-untyped-def]
            if value is None:
                return None
            return "[" + ",".join(str(float(v)) for v in value) + "]"

        return process

    def result_processor(self, dialect, coltype):  # type: ignore[no-untyped-def]
        def process(value):  # type: ignore[no-untyped-def]
            if value is None:
                return None
            if isinstance(value, str):
                return [float(v) for v in value[1:-1].split(",")]
            return list(value)

        return process
DIMS = get_settings().VECTOR_DIMS  # 2048 (Nemotron-3-Embed-1B)


def _now() -> datetime:
    return datetime.now(UTC)


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    doc_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    revision_id: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    fts: Mapped[str] = mapped_column(
        TSVECTOR(),
        Computed("to_tsvector('english', text)", persisted=True),
        nullable=False,
        info={"note": "BM25-ish lexical leg; maintained automatically, never written"},
    )
    chunk_hash: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    embedding: Mapped[list[float]] = mapped_column(HalfVec(DIMS), nullable=False)
    meta: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default=sa_text("'{}'::jsonb"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )

    __table_args__ = (
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": "16", "ef_construction": "64"},
            postgresql_ops={"embedding": "halfvec_cosine_ops"},
        ),
        Index("ix_chunks_fts", "fts", postgresql_using="gin"),
    )


class ConversationThread(Base):
    __tablename__ = "conversation_threads"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    channel: Mapped[str] = mapped_column(Text, nullable=False)
    sender_id: Mapped[str] = mapped_column(Text, nullable=False)  # wa_id for WhatsApp
    title: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    summary: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    summary_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now, index=True
    )

    __table_args__ = (UniqueConstraint("channel", "sender_id", name="uq_threads_channel_sender"),)


class ThreadMessage(Base):
    __tablename__ = "thread_messages"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    thread_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversation_threads.id", ondelete="CASCADE"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    external_id: Mapped[str | None] = mapped_column(Text, unique=True)  # Meta message.id; NULLs OK
    tokens: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )

    __table_args__ = (
        CheckConstraint("role IN ('user','assistant','system')", name="ck_messages_role"),
        Index("ix_messages_thread_created", "thread_id", "created_at"),
    )


class CacheEntry(Base):
    __tablename__ = "cache_entries"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_hash: Mapped[str] = mapped_column(Text, nullable=False, unique=True)  # Tier-1 KV key
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    query_emb: Mapped[list[float]] = mapped_column(HalfVec(DIMS), nullable=False)  # Tier-2
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, server_default="{}")
    doc_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    revision_id: Mapped[str] = mapped_column(Text, nullable=False)
    hit_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    last_hit_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )

    __table_args__ = (
        Index(
            "ix_cache_query_emb_hnsw",
            "query_emb",
            postgresql_using="hnsw",
            postgresql_with={"m": "16", "ef_construction": "64"},
            postgresql_ops={"query_emb": "halfvec_cosine_ops"},
        ),
    )
