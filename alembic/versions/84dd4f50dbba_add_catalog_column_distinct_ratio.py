# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""add catalog_column distinct_ratio

Revision ID: 84dd4f50dbba
Revises: 7d864cacef9c
Create Date: 2026-11-12 14:10:00.000000

"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "84dd4f50dbba"
down_revision: str | Sequence[str] | None = "7d864cacef9c"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "catalog_column", sa.Column("distinct_ratio", sa.Float(), nullable=True)
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("catalog_column", "distinct_ratio")
