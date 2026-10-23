# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""add junction table semantics

Revision ID: a11e76ff0053
Revises: 99d4d343dd0b
Create Date: 2026-10-23 20:26:03.997783

"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a11e76ff0053"
down_revision: str | Sequence[str] | None = "99d4d343dd0b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "catalog_table",
        sa.Column("junction_table_kind", sa.Text(), nullable=True),
    )
    op.add_column(
        "catalog_table",
        sa.Column("junction_table_rationale", sa.Text(), nullable=True),
    )
    op.create_check_constraint(
        op.f("ck_catalog_table_junction_table_kind_value"),
        "catalog_table",
        "junction_table_kind IS NULL OR "
        "junction_table_kind IN ('bridge', 'association')",
    )
    op.execute(
        """
        UPDATE catalog_table
        SET junction_table_kind = 'bridge'
        WHERE is_junction_table
        """
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint(
        op.f("ck_catalog_table_junction_table_kind_value"),
        "catalog_table",
        type_="check",
    )
    op.drop_column("catalog_table", "junction_table_rationale")
    op.drop_column("catalog_table", "junction_table_kind")
