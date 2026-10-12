# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""add owner_id

Revision ID: f3a91c0e5b27
Revises: e4850fd803d6
Create Date: 2026-10-12 11:30:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "f3a91c0e5b27"
down_revision: Union[str, Sequence[str], None] = "e4850fd803d6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

#: Database, schema, table, term, and sql attribute.
_TABLES = (
    "catalog_database",
    "catalog_schema",
    "catalog_table",
    "term",
    "sql_attribute",
)


def upgrade() -> None:
    """Upgrade schema."""
    for table in _TABLES:
        op.add_column(table, sa.Column("owner_id", sa.Text(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    for table in reversed(_TABLES):
        op.drop_column(table, "owner_id")
