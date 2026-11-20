# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""add term generation status

Records whether compilation's model produced a Term or a deterministic
placeholder stood in for a failed model step (#533). Existing Terms are
marked ``generated``: placeholders stored before this revision can't be told
apart from real Terms, and the next compile of their table corrects them.

Revision ID: a7c3e9f1d204
Revises: 0b6f3c2a9d14
Create Date: 2026-11-19 10:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a7c3e9f1d204"
down_revision: Union[str, Sequence[str], None] = "0b6f3c2a9d14"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "term",
        sa.Column(
            "generation_status",
            sa.Text(),
            server_default=sa.text("'generated'"),
            nullable=False,
        ),
    )
    op.add_column("term", sa.Column("generation_note", sa.Text(), nullable=True))
    # op.f: the name already follows METADATA's naming convention, which
    # alembic/env.py applies, so it must not be prefixed a second time.
    op.create_check_constraint(
        op.f("ck_term_generation_status_value"),
        "term",
        "generation_status IN ('generated', 'fallback')",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint(op.f("ck_term_generation_status_value"), "term", type_="check")
    op.drop_column("term", "generation_note")
    op.drop_column("term", "generation_status")
