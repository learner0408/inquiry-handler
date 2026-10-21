"""Lexical retrieval leg: Postgres FTS (ts_rank over the generated fts column).

BM25-ish scoring via Postgres full-text ranking; matches the HNSW dense leg on
the same chunk rows. Turning GIN query is websearch_to_tsquery (forgiving of
unquoted user input from chat).
"""

from sqlalchemy import text as sa_text

from src.db.engine import get_sync_engine
from src.rag.vectordb.chunks import ScoredChunk


class LexicalRetriever:
    def retrieve(self, query: str, top_k: int) -> list[ScoredChunk]:
        engine = get_sync_engine()
        with engine.connect() as conn:
            result = conn.execute(
                sa_text(
                    "SELECT doc_id, revision_id, chunk_index, text, "
                    "ts_rank(fts, websearch_to_tsquery('english', :q)) AS score "
                    "FROM chunks "
                    "WHERE fts @@ websearch_to_tsquery('english', :q) "
                    "ORDER BY score DESC, created_at LIMIT :k"
                ),
                {"q": query, "k": top_k},
            )
            # ts_rank has no "distance" semantics; distance carries the score.
            return [ScoredChunk(*row) for row in result.all()]
