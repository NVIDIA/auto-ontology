# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Add description and ingest status to unstructured documents.

Revision ID: a9d3e7c51f20
Revises: c8518dcfaddc
Create Date: 2026-10-22
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "a9d3e7c51f20"
down_revision = "c8518dcfaddc"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "unstructured_document",
        sa.Column("description", sa.Text, nullable=True),
    )
    # Documents that predate this column were ingested whole by the CLI.
    op.add_column(
        "unstructured_document",
        sa.Column("status", sa.Text, nullable=False, server_default="ready"),
    )
    op.add_column(
        "unstructured_document",
        sa.Column("error", sa.Text, nullable=True),
    )
    # Bare name: NAMING_CONVENTION prepends `ck_unstructured_document_`.
    op.create_check_constraint(
        "status_value",
        "unstructured_document",
        "status IN ('processing', 'extracting', 'ready', 'failed')",
    )


def downgrade() -> None:
    op.drop_constraint("status_value", "unstructured_document", type_="check")
    op.drop_column("unstructured_document", "error")
    op.drop_column("unstructured_document", "status")
    op.drop_column("unstructured_document", "description")
