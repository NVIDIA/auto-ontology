# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Merge semantic FK join path default into attribute relations

Empty by design.  ``feature/unstructured`` and ``main`` each added a revision on
top of a different parent, leaving two heads, and ``alembic upgrade head`` is
ambiguous while that is true.  This revision only rejoins the two lineages; both
sides' DDL has already run by the time it is reached.

Revision ID: c8518dcfaddc
Revises: 99d4d343dd0b, f7a1d2e94b63
Create Date: 2026-10-20 13:14:43.367975

"""

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "c8518dcfaddc"
down_revision: Union[str, Sequence[str], None] = ("99d4d343dd0b", "f7a1d2e94b63")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
