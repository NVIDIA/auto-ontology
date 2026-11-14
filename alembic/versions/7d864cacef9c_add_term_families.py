# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""add term families

Revision ID: 7d864cacef9c
Revises: 99d4d343dd0b
Create Date: 2026-11-12 13:38:25.033446

"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "7d864cacef9c"
down_revision: str | Sequence[str] | None = "99d4d343dd0b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "term_family",
        sa.Column(
            "id",
            sa.Text(),
            server_default=sa.text("(gen_random_uuid())::text"),
            nullable=False,
        ),
        sa.Column("schema_id", sa.Text(), nullable=False),
        sa.Column("prefix", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["schema_id"],
            ["catalog_schema.id"],
            name=op.f("fk_term_family_schema_id_catalog_schema"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_term_family")),
        sa.UniqueConstraint("schema_id", "prefix", name="uq_term_family_schema_prefix"),
    )
    op.create_table(
        "term__term_family",
        sa.Column("term_id", sa.Text(), nullable=False),
        sa.Column("family_id", sa.Text(), nullable=False),
        sa.Column("base_name", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["family_id"],
            ["term_family.id"],
            name=op.f("fk_term__term_family_family_id_term_family"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["term_id"],
            ["term.id"],
            name=op.f("fk_term__term_family_term_id_term"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "term_id", "family_id", name=op.f("pk_term__term_family")
        ),
        sa.UniqueConstraint("term_id", name=op.f("uq_term__term_family_term_id")),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("term__term_family")
    op.drop_table("term_family")
