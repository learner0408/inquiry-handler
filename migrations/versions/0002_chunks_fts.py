"""chunks full-text search: generated tsvector column + GIN index (BM25-ish leg).

Revision ID: 0002_chunks_fts
Revises: 0001_baseline
Create Date: 2026-10-16
"""

import sqlalchemy as sa
from alembic import op

revision = "0002_chunks_fts"
down_revision = "0001_baseline"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "chunks",
        sa.Column(
            "fts",
            sa.dialects.postgresql.TSVECTOR(),
            sa.Computed("to_tsvector('english', text)", persisted=True),
            nullable=False,
        ),
    )
    op.create_index("ix_chunks_fts", "chunks", ["fts"], postgresql_using="gin")


def downgrade() -> None:
    op.drop_index("ix_chunks_fts", table_name="chunks")
    op.drop_column("chunks", "fts")
