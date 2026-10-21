"""Chunk-embedding persistence: VectorDBInterface + pgvector impl.

Atomic per-doc replace (agreed staleness contract): delete rows of this doc
with a different revision FIRST, then insert new rows (conflicts on chunk_hash
ignored). NOTE: chunk_hash is text-scoped, so byte-identical text shared by two
docs shares one row (cited to the first doc) — rare in practice, revisit if
multi-tenant docs collide.
"""

from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import text as sa_text
from sqlalchemy.dialects.postgresql import insert

from src.db.engine import get_sync_engine
from src.db.models import Chunk


@dataclass
class ChunkRow:
    chunk_index: int
    text: str
    chunk_hash: str
    embedding: list[float]


@dataclass
class ScoredChunk:
    doc_id: str
    revision_id: str
    chunk_index: int
    text: str
    distance: float


class VectorDBInterface(Protocol):
    def atomic_replace(self, doc_id: str, revision_id: str, rows: list[ChunkRow]) -> dict: ...
    def search(self, query_emb: list[float], top_k: int) -> list[ScoredChunk]: ...


class PgVectorChunks:
    def atomic_replace(self, doc_id: str, revision_id: str, rows: list[ChunkRow]) -> dict:
        engine = get_sync_engine()
        with engine.begin() as conn:
            deleted = conn.execute(
                sa_text("DELETE FROM chunks WHERE doc_id = :d AND revision_id != :r"),
                {"d": doc_id, "r": revision_id},
            ).rowcount
            before = conn.execute(
                sa_text("SELECT COUNT(*) FROM chunks WHERE doc_id = :d"),
                {"d": doc_id},
            ).scalar_one()
            inserted = 0
            if rows:
                conn.execute(
                    insert(Chunk).values(
                        [
                            {
                                "doc_id": doc_id,
                                "revision_id": revision_id,
                                "chunk_index": r.chunk_index,
                                "text": r.text,
                                "chunk_hash": r.chunk_hash,
                                "embedding": r.embedding,
                            }
                            for r in rows
                        ]
                    ).on_conflict_do_nothing(index_elements=["chunk_hash"])
                )
                # rowcount is unreliable for ON CONFLICT DO NOTHING; diff instead.
                after = conn.execute(
                    sa_text("SELECT COUNT(*) FROM chunks WHERE doc_id = :d"),
                    {"d": doc_id},
                ).scalar_one()
                inserted = after - before
        return {"doc_id": doc_id, "revision_id": revision_id, "inserted": inserted,
                "deleted": deleted, "chunks": len(rows)}

    def search(self, query_emb: list[float], top_k: int) -> list[ScoredChunk]:
        literal = "[" + ",".join(str(float(v)) for v in query_emb) + "]"
        engine = get_sync_engine()
        with engine.connect() as conn:
            result = conn.execute(
                sa_text(
                    "SELECT doc_id, revision_id, chunk_index, text, "
                    "embedding <=> CAST(:qv AS halfvec) AS distance "
                    "FROM chunks ORDER BY distance LIMIT :k"
                ),
                {"qv": literal, "k": top_k},
            )
            return [ScoredChunk(*row) for row in result.all()]
