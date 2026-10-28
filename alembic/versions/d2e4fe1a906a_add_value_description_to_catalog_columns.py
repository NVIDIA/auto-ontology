# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""add value description to catalog columns

Revision ID: d2e4fe1a906a
Revises: 99d4d343dd0b
Create Date: 2026-10-28 22:30:00.000000

"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "d2e4fe1a906a"
down_revision: str | Sequence[str] | None = "99d4d343dd0b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "catalog_column", sa.Column("value_description", sa.Text(), nullable=True)
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("catalog_column", "value_description")
