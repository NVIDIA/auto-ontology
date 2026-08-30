"""Add semantic_compilation_history

Revision ID: 9f1f77724292
Revises: 859d45db5f8a
Create Date: 2026-08-30 20:53:23.085688

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9f1f77724292'
down_revision: Union[str, Sequence[str], None] = '859d45db5f8a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # `if_not_exists`, because this table predates its own migration: it has
    # been created at service startup by `ensure_history_table` in
    # `gsf/ingestion_service/history.py` since it was introduced, so every
    # environment already running that code has it. A plain CREATE would fail
    # there on "relation already exists" and block the upgrade.
    #
    # The column definitions below match that CREATE exactly. They are declared
    # in `gsf/dal/schema.py` now as well, which is what stops autogenerate from
    # reading the table as drift and proposing `drop_table` for it.
    op.create_table('semantic_compilation_history',
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('started_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('status', sa.Text(), nullable=True),
    sa.CheckConstraint("status IS NULL OR status IN ('succeeded', 'failed')", name=op.f('ck_semantic_compilation_history_status_value')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_semantic_compilation_history')),
    if_not_exists=True
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('semantic_compilation_history')
