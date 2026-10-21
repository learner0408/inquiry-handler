"""Semantic cache checked BEFORE vectordb (Tier-1 exact, Tier-2 HNSW @0.95).

Tier-1: sha256(normalized query) exact KV on cache_entries.query_hash.
Tier-2: cosine over query_emb; hit iff distance <= 1 - CACHE_SIM_THRESHOLD and
the entry's revision still exists for its doc_id in chunks (per-doc staleness;
atomic-replace keeps exactly one revision per doc). set() refuses blank answers
(never cache fallbacks) and prunes same-doc rows with a different revision in
the same transaction (invalidate on revision bump).
"""

import hashlib
import re
import uuid
from dataclasses import dataclass

from sqlalchemy import text as sa_text

from src.config import get_settings
from src.db.engine import get_sync_engine
from src.rag.embeddings.embedder import embed_query


@dataclass
class CacheHit:
    query_hash: str
    query_text: str
    answer: str
    citations: list[str]
    revision_id: str


def query_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _vec_literal(vec: list[float]) -> str:
    return "[" + ",".join(str(float(v)) for v in vec) + "]"


class SemanticCache:
    def get(self, query_text: str) -> CacheHit | None:
        settings = get_settings()
        max_cosine_distance = 1 - settings.CACHE_SIM_THRESHOLD
        engine = get_sync_engine()
        with engine.connect() as conn:
            # Freshness is per-entry (its revision still current for its doc);
            # a single caller-passed revision is ambiguous across documents.
            # Tier-1: exact normalized-question match
            row = conn.execute(
                sa_text(
                    "SELECT query_text, answer, citations, revision_id "
                    "FROM cache_entries c "
                    "WHERE query_hash = :h "
                    "AND EXISTS (SELECT 1 FROM chunks k "
                    "            WHERE k.doc_id = c.doc_id AND k.revision_id = c.revision_id)"
                ),
                {"h": query_hash(query_text)},
            ).first()
            tier = 1
            if row is None:
                # Tier-2: closest query_emb among CURRENT-revision entries only
                row = conn.execute(
                    sa_text(
                        "SELECT c.query_hash, c.query_text, c.answer, c.citations, "
                        "c.revision_id, c.query_emb <=> CAST(:qv AS halfvec) AS distance "
                        "FROM cache_entries c "
                        "WHERE EXISTS (SELECT 1 FROM chunks k "
                        "            WHERE k.doc_id = c.doc_id AND k.revision_id = c.revision_id) "
                        "  AND c.query_emb <=> CAST(:qv AS halfvec) <= :maxd "
                        "ORDER BY distance LIMIT 1"
                    ),
                    {"qv": _vec_literal(embed_query(query_text)),
                     "maxd": max_cosine_distance},
                ).first()
                tier = 2
            if row is None:
                return None
            hit = CacheHit(
                query_hash=query_hash(query_text) if tier == 1 else row.query_hash,
                query_text=row.query_text,
                answer=row.answer,
                citations=list(row.citations),
                revision_id=row.revision_id,
            )
            with engine.begin() as bump:
                bump.execute(
                    sa_text(
                        "UPDATE cache_entries SET hit_count = hit_count + 1, "
                        "last_hit_at = now() WHERE query_hash = :h"
                    ),
                    {"h": hit.query_hash},
                )
            return hit

    def set(
        self,
        query_text: str,
        revision_id: str,
        answer: str,
        citations: list[str],
        doc_id: str,
    ) -> CacheHit:
        if not answer.strip():
            raise ValueError("refusing to cache a blank answer (fallbacks are never cached)")
        h = query_hash(query_text)
        engine = get_sync_engine()
        with engine.begin() as conn:
            # invalidate-on-bump: purge same doc carrying any other revision
            conn.execute(
                sa_text("DELETE FROM cache_entries WHERE doc_id = :d AND revision_id != :r"),
                {"d": doc_id, "r": revision_id},
            )
            conn.execute(
                sa_text(
                    "INSERT INTO cache_entries "
                    "(id, query_hash, query_text, query_emb, answer, citations, "
                    "doc_id, revision_id, created_at) "
                    "VALUES (:id, :h, :qt, CAST(:qv AS halfvec), :a, :c, :d, :r, now()) "
                    "ON CONFLICT (query_hash) DO UPDATE SET "
                    "query_text = EXCLUDED.query_text, query_emb = EXCLUDED.query_emb, "
                    "answer = EXCLUDED.answer, citations = EXCLUDED.citations, "
                    "doc_id = EXCLUDED.doc_id, revision_id = EXCLUDED.revision_id, "
                    "hit_count = 0"
                ),
                {"id": str(uuid.uuid4()), "h": h, "qt": query_text,
                 "qv": _vec_literal(embed_query(query_text)), "a": answer,
                 "c": citations, "d": doc_id, "r": revision_id},
            )
        return CacheHit(query_hash=h, query_text=query_text, answer=answer,
                        citations=citations, revision_id=revision_id)
