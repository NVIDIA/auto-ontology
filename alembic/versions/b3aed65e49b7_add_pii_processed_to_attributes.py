# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""add pii processed to attributes

Revision ID: b3aed65e49b7
Revises: 6347ede6dac1
Create Date: 2026-10-04 15:27:12.272504

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "b3aed65e49b7"
down_revision: Union[str, Sequence[str], None] = "6347ede6dac1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "column_attribute",
        sa.Column(
            "pii_processed",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
    )
    op.add_column(
        "sql_attribute",
        sa.Column(
            "pii_processed",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("sql_attribute", "pii_processed")
    op.drop_column("column_attribute", "pii_processed")
