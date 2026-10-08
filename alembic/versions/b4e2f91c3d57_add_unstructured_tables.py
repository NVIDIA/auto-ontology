# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Add unstructured document/chunk/term/attribute tables.

Revision ID: b4e2f91c3d57
Revises: c778a7d94c20
Create Date: 2026-10-08
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "b4e2f91c3d57"
down_revision = "c778a7d94c20"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "unstructured_document",
        sa.Column(
            "id",
            sa.Text,
            primary_key=True,
            server_default=sa.text("(gen_random_uuid())::text"),
        ),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("path", sa.Text, nullable=False),
        sa.Column("file_type", sa.Text, nullable=False),
        sa.Column("summary", sa.Text, nullable=True),
        sa.Column("publication_date", sa.Text, nullable=True),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint("path", name="uq_unstructured_document_path"),
    )

    op.create_table(
        "unstructured_chunk",
        sa.Column(
            "id",
            sa.Text,
            primary_key=True,
            server_default=sa.text("(gen_random_uuid())::text"),
        ),
        sa.Column(
            "document_id",
            sa.Text,
            sa.ForeignKey("unstructured_document.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("content_type", sa.Text, nullable=False),
        sa.Column("page_number", sa.Integer, nullable=True),
        sa.Column("chunk_index", sa.Integer, nullable=False),
        sa.Column("source_metadata", postgresql.JSONB, nullable=True),
    )
    op.create_index("ix_unstructured_chunk_document_id", "unstructured_chunk", ["document_id"])

    op.create_table(
        "unstructured_term",
        sa.Column(
            "id",
            sa.Text,
            primary_key=True,
            server_default=sa.text("(gen_random_uuid())::text"),
        ),
        sa.Column(
            "document_id",
            sa.Text,
            sa.ForeignKey("unstructured_document.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("description", sa.Text, nullable=True),
    )
    op.create_index("ix_unstructured_term_document_id", "unstructured_term", ["document_id"])

    op.create_table(
        "unstructured_attribute",
        sa.Column(
            "id",
            sa.Text,
            primary_key=True,
            server_default=sa.text("(gen_random_uuid())::text"),
        ),
        sa.Column(
            "term_id",
            sa.Text,
            sa.ForeignKey("unstructured_term.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("attribute_type", sa.Text, nullable=False),
        sa.CheckConstraint("attribute_type IN ('sql', 'ref')", name="ck_unstructured_attribute_attribute_type_value"),
    )
    op.create_index("ix_unstructured_attribute_term_id", "unstructured_attribute", ["term_id"])

    op.create_table(
        "unstructured_attribute__chunk",
        sa.Column(
            "attribute_id",
            sa.Text,
            sa.ForeignKey("unstructured_attribute.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "chunk_id",
            sa.Text,
            sa.ForeignKey("unstructured_chunk.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("score", sa.Text, nullable=True),
    )
    op.create_index(
        "ix_unstructured_attribute__chunk_chunk_id",
        "unstructured_attribute__chunk",
        ["chunk_id"],
    )


def downgrade() -> None:
    op.drop_table("unstructured_attribute__chunk")
    op.drop_table("unstructured_attribute")
    op.drop_table("unstructured_term")
    op.drop_table("unstructured_chunk")
    op.drop_table("unstructured_document")
