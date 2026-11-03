# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""add catalog_table legacy flags

Revision ID: 08de5bc4b3c3
Revises: 99d4d343dd0b
Create Date: 2026-11-03 19:44:59.391564

"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "08de5bc4b3c3"
down_revision: str | Sequence[str] | None = "99d4d343dd0b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "catalog_table",
        sa.Column(
            "is_legacy", sa.Boolean(), server_default=sa.text("false"), nullable=False
        ),
    )
    op.add_column(
        "catalog_table",
        sa.Column("legacy_replacement_table_id", sa.Text(), nullable=True),
    )
    op.add_column("catalog_table", sa.Column("legacy_reason", sa.Text(), nullable=True))
    op.create_foreign_key(
        op.f("fk_catalog_table_legacy_replacement_table_id_catalog_table"),
        "catalog_table",
        "catalog_table",
        ["legacy_replacement_table_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint(
        op.f("fk_catalog_table_legacy_replacement_table_id_catalog_table"),
        "catalog_table",
        type_="foreignkey",
    )
    op.drop_column("catalog_table", "legacy_reason")
    op.drop_column("catalog_table", "legacy_replacement_table_id")
    op.drop_column("catalog_table", "is_legacy")
