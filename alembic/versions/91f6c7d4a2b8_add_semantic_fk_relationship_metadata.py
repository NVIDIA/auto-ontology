# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""add semantic fk relationship metadata

Revision ID: 91f6c7d4a2b8
Revises: f3a91c0e5b27
Create Date: 2026-10-19 16:30:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "91f6c7d4a2b8"
down_revision: Union[str, Sequence[str], None] = "f3a91c0e5b27"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add the versioned metadata envelope to semantic FK edges."""
    op.add_column(
        "column__semantic_fk",
        sa.Column(
            "relationship_metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    """Remove semantic FK relationship metadata."""
    op.drop_column("column__semantic_fk", "relationship_metadata")
