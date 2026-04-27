"""add_session_summary_field

Revision ID: 8c4d2a1e9f00
Revises: 52b0e0cdd01d
Create Date: 2026-04-27

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "8c4d2a1e9f00"
down_revision: Union[str, Sequence[str], None] = "52b0e0cdd01d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
  op.add_column(
      "rag_bot_sessions",
      sa.Column("summary", sa.Text(), nullable=True),
  )


def downgrade() -> None:
  op.drop_column("rag_bot_sessions", "summary")
