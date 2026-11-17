# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""add semantic fk join path default

Revision ID: 99d4d343dd0b
Revises: f3a91c0e5b27
Create Date: 2026-10-20 00:30:15.515353

"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "99d4d343dd0b"
down_revision: str | Sequence[str] | None = "f3a91c0e5b27"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "column__semantic_fk",
        sa.Column(
            "use_in_join_paths",
            sa.Boolean(),
            server_default=sa.text("true"),
            nullable=False,
        ),
    )
    op.execute(
        """
        WITH ranked AS (
            SELECT
                semantic_fk.column_id,
                semantic_fk.attribute_id,
                row_number() OVER (
                    PARTITION BY source_column.table_id, semantic_fk.attribute_id
                    ORDER BY lower(source_column.name), source_column.id
                ) AS rank
            FROM column__semantic_fk AS semantic_fk
            JOIN catalog_column AS source_column
              ON source_column.id = semantic_fk.column_id
        )
        UPDATE column__semantic_fk AS semantic_fk
        SET use_in_join_paths = (ranked.rank = 1)
        FROM ranked
        WHERE semantic_fk.column_id = ranked.column_id
          AND semantic_fk.attribute_id = ranked.attribute_id
        """
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("column__semantic_fk", "use_in_join_paths")
