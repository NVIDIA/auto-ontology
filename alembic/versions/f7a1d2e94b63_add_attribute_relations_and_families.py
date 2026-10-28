# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Relate attributes to each other, and group the ones measuring the same thing.

Revision ID: f7a1d2e94b63
Revises: e3c6a0f41b82
Create Date: 2026-10-14
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "f7a1d2e94b63"
down_revision = "e3c6a0f41b82"
branch_labels = None
depends_on = None

RELATIONS = (
    "same_measure_different_place",
    "same_measure_different_period",
    "same_measure_different_source",
    "aggregate_and_part",
    "related_but_distinct",
    "inverse",
    "unrelated",
)


def upgrade() -> None:
    op.create_table(
        "unstructured_attribute__attribute",
        sa.Column("attribute_id_a", sa.Text, primary_key=True),
        sa.Column("attribute_id_b", sa.Text, primary_key=True),
        sa.Column("relation", sa.Text, nullable=False),
        sa.Column("canonical", sa.Text, nullable=True),
        sa.Column("reason", sa.Text, nullable=True),
        sa.Column("qualifier_a", sa.Text, nullable=True),
        sa.Column("qualifier_b", sa.Text, nullable=True),
        sa.Column("aggregate_side", sa.Text, nullable=True),
        sa.Column("distance", sa.Float, nullable=True),
        sa.Column("model", sa.Text, nullable=False),
        sa.ForeignKeyConstraint(
            ["attribute_id_a"], ["unstructured_attribute.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["attribute_id_b"], ["unstructured_attribute.id"], ondelete="CASCADE"
        ),
        # Bare names, no `ck_<table>_` prefix: NAMING_CONVENTION prepends it, and
        # spelling it out here gets it applied twice -- past 63 characters, which
        # Postgres truncates to a hash the model never matches again.
        sa.CheckConstraint(
            "relation IN (" + ", ".join(f"'{r}'" for r in RELATIONS) + ")",
            name="relation_value",
        ),
        sa.CheckConstraint(
            "aggregate_side IS NULL OR aggregate_side IN ('a', 'b')",
            name="aggregate_side_value",
        ),
        # One row per unordered pair: the relations are symmetric apart from
        # aggregate_and_part, which names its whole in aggregate_side instead
        # of relying on column order.
        sa.CheckConstraint(
            "attribute_id_a < attribute_id_b",
            name="ordered",
        ),
    )
    op.create_index(
        "ix_unstructured_attribute__attribute_b",
        "unstructured_attribute__attribute",
        ["attribute_id_b"],
    )

    op.create_table(
        "unstructured_attribute_family",
        sa.Column(
            "id",
            sa.Text,
            primary_key=True,
            server_default=sa.text("(gen_random_uuid())::text"),
        ),
        sa.Column("canonical", sa.Text, nullable=False),
        sa.Column("dimension", sa.Text, nullable=True),
        sa.Column("note", sa.Text, nullable=True),
        sa.Column("model", sa.Text, nullable=False),
        sa.CheckConstraint(
            "dimension IS NULL OR dimension IN ('place', 'period', 'source')",
            name="dimension_value",
        ),
    )

    op.create_table(
        "unstructured_attribute_family__attribute",
        sa.Column("family_id", sa.Text, primary_key=True),
        sa.Column("attribute_id", sa.Text, primary_key=True),
        sa.Column("qualifier", sa.Text, nullable=True),
        sa.Column("value", sa.Text, nullable=True),
        sa.ForeignKeyConstraint(
            ["family_id"], ["unstructured_attribute_family.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["attribute_id"], ["unstructured_attribute.id"], ondelete="CASCADE"
        ),
    )
    op.create_index(
        "ix_unstructured_attribute_family__attribute_attribute_id",
        "unstructured_attribute_family__attribute",
        ["attribute_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_unstructured_attribute_family__attribute_attribute_id",
        table_name="unstructured_attribute_family__attribute",
    )
    op.drop_table("unstructured_attribute_family__attribute")
    op.drop_table("unstructured_attribute_family")
    op.drop_index(
        "ix_unstructured_attribute__attribute_b",
        table_name="unstructured_attribute__attribute",
    )
    op.drop_table("unstructured_attribute__attribute")
