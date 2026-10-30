# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""add ambiguity_resolver to column attributes

Revision ID: a831a6fb92ae
Revises: d2e4fe1a906a
Create Date: 2026-10-30 10:12:01.593156

"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a831a6fb92ae"
down_revision: str | Sequence[str] | None = "d2e4fe1a906a"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "column_attribute",
        sa.Column(
            "ambiguity_resolver",
            sa.Text(),
            server_default=sa.text("''"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("column_attribute", "ambiguity_resolver")
