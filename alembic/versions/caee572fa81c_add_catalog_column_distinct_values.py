# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""add catalog_column distinct_values

Revision ID: caee572fa81c
Revises: 84dd4f50dbba
Create Date: 2026-11-12 15:06:00.000000

"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "caee572fa81c"
down_revision: str | Sequence[str] | None = "84dd4f50dbba"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "catalog_column", sa.Column("distinct_values", sa.Text(), nullable=True)
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("catalog_column", "distinct_values")
