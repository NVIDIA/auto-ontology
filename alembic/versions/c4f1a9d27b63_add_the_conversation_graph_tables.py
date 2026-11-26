# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""add the conversation graph tables

``conversation_node`` with the five kinds it started with; ``d2b6f4a81e05``
widens them.

Revision ID: c4f1a9d27b63
Revises: a1c7e3f20b94
Create Date: 2026-10-20 10:15:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c4f1a9d27b63"
down_revision: Union[str, Sequence[str], None] = "a1c7e3f20b94"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "conversation_node",
        sa.Column(
            "id",
            sa.Text(),
            server_default=sa.text("(gen_random_uuid())::text"),
            nullable=False,
        ),
        sa.Column("database_id", sa.Text(), nullable=False),
        sa.Column("conversation_id", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("normalized_name", sa.Text(), nullable=False),
        sa.Column("type", sa.Text(), nullable=False),
        sa.Column("kinds", sa.ARRAY(sa.Text()), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("attribute_key", sa.Text(), nullable=True),
        sa.Column("value", sa.Text(), nullable=True),
        sa.Column("turn_indexes", sa.ARRAY(sa.Integer()), nullable=True),
        sa.Column("evidence", sa.Text(), nullable=True),
        sa.Column("time_anchor", sa.Text(), nullable=True),
        sa.Column("occurred_on", sa.Date(), nullable=True),
        sa.Column(
            "created",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.Column("modified", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_by", sa.Text(), nullable=True),
        sa.Column("modified_by", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "kinds <@ ARRAY['object', 'fact', 'preference', 'event', 'state']::text[]"
            " AND array_length(kinds, 1) >= 1",
            name=op.f("ck_conversation_node_conversation_node_kinds_value"),
        ),
        sa.ForeignKeyConstraint(
            ["database_id"],
            ["catalog_database.id"],
            name=op.f("fk_conversation_node_database_id_catalog_database"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_conversation_node")),
        sa.UniqueConstraint(
            "database_id",
            "conversation_id",
            "normalized_name",
            name="uq_conversation_node_name",
        ),
    )
    op.create_index(
        "ix_conversation_node_database_conversation",
        "conversation_node",
        ["database_id", "conversation_id"],
        unique=False,
    )
    op.create_index(
        "ix_conversation_node_kinds",
        "conversation_node",
        ["kinds"],
        unique=False,
        postgresql_using="gin",
    )
    op.create_table(
        "conversation_edge",
        sa.Column("source_id", sa.Text(), nullable=False),
        sa.Column("target_id", sa.Text(), nullable=False),
        sa.Column("type", sa.Text(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("statement", sa.Text(), nullable=True),
        sa.Column("turn_indexes", sa.ARRAY(sa.Integer()), nullable=True),
        sa.Column("evidence", sa.Text(), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column(
            "derived", sa.Boolean(), server_default=sa.text("false"), nullable=False
        ),
        sa.Column("distance", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "type <> 'semantic' OR role IN "
            "('supersedes', 'contradicts', 'precedes', 'refines', 'unrelated')",
            name=op.f("ck_conversation_edge_conversation_edge_semantic_role_value"),
        ),
        sa.CheckConstraint(
            "type IN ('knowledge', 'semantic')",
            name=op.f("ck_conversation_edge_conversation_edge_type_value"),
        ),
        sa.ForeignKeyConstraint(
            ["source_id"],
            ["conversation_node.id"],
            name=op.f("fk_conversation_edge_source_id_conversation_node"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["target_id"],
            ["conversation_node.id"],
            name=op.f("fk_conversation_edge_target_id_conversation_node"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "source_id", "target_id", "type", "role", name=op.f("pk_conversation_edge")
        ),
    )
    op.create_index(
        "ix_conversation_edge_source",
        "conversation_edge",
        ["source_id"],
        unique=False,
    )
    op.create_index(
        "ix_conversation_edge_target",
        "conversation_edge",
        ["target_id"],
        unique=False,
    )
    op.create_index(
        "ix_conversation_edge_type",
        "conversation_edge",
        ["type"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_conversation_edge_type", table_name="conversation_edge")
    op.drop_index("ix_conversation_edge_target", table_name="conversation_edge")
    op.drop_index("ix_conversation_edge_source", table_name="conversation_edge")
    op.drop_table("conversation_edge")
    op.drop_index("ix_conversation_node_kinds", table_name="conversation_node")
    op.drop_index(
        "ix_conversation_node_database_conversation", table_name="conversation_node"
    )
    op.drop_table("conversation_node")
