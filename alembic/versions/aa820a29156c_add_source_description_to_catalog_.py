# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""add source description to catalog tables and columns

Revision ID: aa820a29156c
Revises: 99d4d343dd0b
Create Date: 2026-10-28 17:26:41.999908

"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "aa820a29156c"
down_revision: str | Sequence[str] | None = "99d4d343dd0b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "catalog_column", sa.Column("source_description", sa.Text(), nullable=True)
    )
    op.add_column(
        "catalog_table", sa.Column("source_description", sa.Text(), nullable=True)
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("catalog_table", "source_description")
    op.drop_column("catalog_column", "source_description")
