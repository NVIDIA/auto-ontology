# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""add user groups

Revision ID: e4850fd803d6
Revises: b3aed65e49b7
Create Date: 2026-10-11 12:39:02.031785

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "e4850fd803d6"
down_revision: Union[str, Sequence[str], None] = "b3aed65e49b7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "user_group",
        sa.Column(
            "id",
            sa.Text(),
            server_default=sa.text("(gen_random_uuid())::text"),
            nullable=False,
        ),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("color", sa.Text(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column(
            "created",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "role IN ('admin', 'viewer')", name=op.f("ck_user_group_role_value")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_user_group")),
    )
    op.create_index(
        "uq_user_group_name_lower",
        "user_group",
        [sa.literal_column("lower(TRIM(BOTH FROM name))")],
        unique=True,
    )
    op.create_table(
        "user_group__member",
        sa.Column("group_id", sa.Text(), nullable=False),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["group_id"],
            ["user_group.id"],
            name=op.f("fk_user_group__member_group_id_user_group"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "group_id", "user_id", name=op.f("pk_user_group__member")
        ),
    )
    op.create_index(
        "ix_user_group__member_user_id",
        "user_group__member",
        ["user_id"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_user_group__member_user_id", table_name="user_group__member")
    op.drop_table("user_group__member")
    op.drop_index("uq_user_group_name_lower", table_name="user_group")
    op.drop_table("user_group")
