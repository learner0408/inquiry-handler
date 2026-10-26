"""conversation threads: batched summary boundary timestamp.

Revision ID: 0003_thread_summary_timestamp
Revises: 0002_chunks_fts
Create Date: 2026-10-24
"""

import sqlalchemy as sa
from alembic import op

revision = "0003_thread_summary_timestamp"
down_revision = "0002_chunks_fts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "conversation_threads",
        sa.Column("summary_timestamp", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("conversation_threads", "summary_timestamp")
