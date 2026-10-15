# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""add zone owner and members

Revision ID: 8b1d6f0c3a47
Revises: f3a91c0e5b27
Create Date: 2026-10-15 15:10:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "8b1d6f0c3a47"
down_revision: Union[str, Sequence[str], None] = "f3a91c0e5b27"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("zone", sa.Column("owner_id", sa.Text(), nullable=True))
    op.create_table(
        "zone__member",
        sa.Column("zone_id", sa.Text(), nullable=False),
        sa.Column("member_id", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["zone_id"],
            ["zone.id"],
            name=op.f("fk_zone__member_zone_id_zone"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("zone_id", "member_id", name=op.f("pk_zone__member")),
    )
    op.create_index(
        "ix_zone__member_member_id",
        "zone__member",
        ["member_id"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_zone__member_member_id", table_name="zone__member")
    op.drop_table("zone__member")
    op.drop_column("zone", "owner_id")
