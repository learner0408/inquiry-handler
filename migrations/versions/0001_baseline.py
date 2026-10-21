"""baseline: matches create_all v0 schema (4 tables); each revision depends on it.

Revision ID: 0001_baseline
Revises:
Create Date: 2026-10-16
"""

import sqlalchemy as sa  # noqa: F401
from alembic import op  # noqa: F401

revision = "0001_baseline"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # v0 schema is created by scripts/init_db.py (create_all); this revision is
    # the stamp point so fresh DBs follow: init_db -> alembic stamp head -> beyond.
    pass


def downgrade() -> None:
    pass
