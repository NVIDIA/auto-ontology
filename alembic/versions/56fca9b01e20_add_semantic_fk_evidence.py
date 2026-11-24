# SPDX-FileCopyrightText: Copyright (c) 2026 workdd
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""add semantic fk evidence

Revision ID: 56fca9b01e20
Revises: caee572fa81c
Create Date: 2026-11-24 10:28:10.778317

"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "56fca9b01e20"
down_revision: str | Sequence[str] | None = "caee572fa81c"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("column__semantic_fk", sa.Column("method", sa.Text(), nullable=True))
    op.add_column(
        "column__semantic_fk", sa.Column("containment", sa.Float(), nullable=True)
    )
    op.add_column(
        "column__semantic_fk", sa.Column("vector_distance", sa.Float(), nullable=True)
    )
    op.add_column(
        "column__semantic_fk",
        sa.Column("join_verified", sa.Boolean(), nullable=True),
    )
    op.add_column(
        "column__semantic_fk", sa.Column("rationale", sa.Text(), nullable=True)
    )
    op.create_check_constraint(
        op.f("ck_column__semantic_fk_method_value"),
        "column__semantic_fk",
        "method IS NULL OR method IN "
        "('declared', 'llm_pick', 'value_confirmed', 'value_match')",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint(
        op.f("ck_column__semantic_fk_method_value"),
        "column__semantic_fk",
        type_="check",
    )
    op.drop_column("column__semantic_fk", "rationale")
    op.drop_column("column__semantic_fk", "join_verified")
    op.drop_column("column__semantic_fk", "vector_distance")
    op.drop_column("column__semantic_fk", "containment")
    op.drop_column("column__semantic_fk", "method")
