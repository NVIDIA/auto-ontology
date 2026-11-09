# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""add declared column uniqueness

Revision ID: a3f6c1d92b47
Revises: 99d4d343dd0b
Create Date: 2026-11-06 23:30:00.000000

"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a3f6c1d92b47"
down_revision: str | Sequence[str] | None = "99d4d343dd0b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Record which columns a source declared unique.

    Nullable, and NULL for every existing row: ingest fills it in, and a
    column nothing has declared keeps the sampled answer it already had.
    """
    op.add_column(
        "catalog_column",
        sa.Column("is_unique_declared", sa.Boolean(), nullable=True),
    )


def downgrade() -> None:
    """Drop the declaration record; ``is_unique`` keeps what it had."""
    op.drop_column("catalog_column", "is_unique_declared")
