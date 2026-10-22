# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record how each attribute -> chunk link was built.

Revision ID: e3c6a0f41b82
Revises: b4e2f91c3d57
Create Date: 2026-10-11
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "e3c6a0f41b82"
down_revision = "b4e2f91c3d57"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Links that predate this column all came from the vector search, which is
    # what the default backfills them to.
    op.add_column(
        "unstructured_attribute__chunk",
        sa.Column("source", sa.Text, nullable=False, server_default="vector"),
    )
    # Bare name: NAMING_CONVENTION prepends `ck_unstructured_attribute__chunk_`
    # on both the create and the drop, so writing the prefix here applies it twice.
    op.create_check_constraint(
        "source_value",
        "unstructured_attribute__chunk",
        "source IN ('llm', 'vector')",
    )


def downgrade() -> None:
    op.drop_constraint(
        "source_value",
        "unstructured_attribute__chunk",
        type_="check",
    )
    op.drop_column("unstructured_attribute__chunk", "source")
