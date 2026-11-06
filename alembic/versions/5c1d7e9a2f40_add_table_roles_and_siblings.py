# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""add table roles, grain and sibling tables

Revision ID: 5c1d7e9a2f40
Revises: 08de5bc4b3c3
Create Date: 2026-11-06 09:00:00.000000

"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "5c1d7e9a2f40"
down_revision: str | Sequence[str] | None = "08de5bc4b3c3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("catalog_table", sa.Column("table_role", sa.Text(), nullable=True))
    op.add_column("catalog_table", sa.Column("grain", sa.Text(), nullable=True))
    op.create_table(
        "table__sibling",
        sa.Column("table_id", sa.Text(), nullable=False),
        sa.Column("sibling_table_id", sa.Text(), nullable=False),
        sa.Column("note", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["table_id"],
            ["catalog_table.id"],
            name=op.f("fk_table__sibling_table_id_catalog_table"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["sibling_table_id"],
            ["catalog_table.id"],
            name=op.f("fk_table__sibling_sibling_table_id_catalog_table"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "table_id", "sibling_table_id", name=op.f("pk_table__sibling")
        ),
    )
    op.create_index(
        "ix_table__sibling_sibling",
        "table__sibling",
        ["sibling_table_id"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_table__sibling_sibling", table_name="table__sibling")
    op.drop_table("table__sibling")
    op.drop_column("catalog_table", "grain")
    op.drop_column("catalog_table", "table_role")
