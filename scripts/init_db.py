"""Create pgvector extension + all tables (v0 baseline) and stamp Alembic head.

Fresh dev DB: init_db.py once, then `alembic upgrade head` for any later
revisions beyond the stamp point (e.g. 0002_chunks_fts is already in create_all).
Usage: uv run python scripts/init_db.py && uv run alembic stamp head
"""

from sqlalchemy import text

# Import models so they register on Base.metadata
from src.db import models  # noqa: F401
from src.db.engine import Base, get_sync_engine


def main() -> None:
    engine = get_sync_engine()
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)
    print("tables:", sorted(Base.metadata.tables.keys()))


if __name__ == "__main__":
    main()
