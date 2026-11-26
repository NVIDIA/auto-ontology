# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Merge term generation and conversation heads

Revision ID: 8d94b61ac2f0
Revises: a7c3e9f1d204, e7a3d5c19f42
Create Date: 2026-11-25 11:27:09.888710

"""

revision: str = "8d94b61ac2f0"
down_revision: tuple[str, str] = ("a7c3e9f1d204", "e7a3d5c19f42")
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None


def upgrade() -> None:
    """Join the two heads without changing the schema."""


def downgrade() -> None:
    """Restore the separate heads without changing the schema."""
