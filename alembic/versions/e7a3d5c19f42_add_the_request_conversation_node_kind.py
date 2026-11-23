# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""add the request conversation_node kind

``request`` is something the speaker asked for help with, as against something
the project has. ``schema.py`` admits it; without this migration a database
upgraded to head rejects every node extracted with it.

Drops both constraint names for the reason ``d2b6f4a81e05`` gives: a database
widened by hand carries the declared name as a second, independently-enforced
check.

Revision ID: e7a3d5c19f42
Revises: d2b6f4a81e05
Create Date: 2026-11-23 14:00:00.000000

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e7a3d5c19f42"
down_revision: Union[str, Sequence[str], None] = "d2b6f4a81e05"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

DECLARED = "conversation_node_kinds_value"
RENDERED = f"ck_conversation_node_{DECLARED}"

BEFORE = ("object", "fact", "preference", "event", "state", "target", "proposed")
AFTER = (*BEFORE, "request")


def _check(kinds: Sequence[str]) -> str:
    values = ", ".join(f"'{kind}'" for kind in kinds)
    return f"kinds <@ ARRAY[{values}]::text[] AND array_length(kinds, 1) >= 1"


def _replace(kinds: Sequence[str]) -> None:
    """Leave exactly one check on ``kinds``, admitting *kinds*."""
    for name in (RENDERED, DECLARED):
        op.execute(f"ALTER TABLE conversation_node DROP CONSTRAINT IF EXISTS {name}")
    op.create_check_constraint(DECLARED, "conversation_node", _check(kinds))


def upgrade() -> None:
    """Upgrade schema."""
    _replace(AFTER)


def downgrade() -> None:
    """Downgrade schema.

    ``request`` is not representable under the narrower check, so it is removed
    from ``kinds``, and only a node left with no kind is deleted.
    """
    op.execute(
        "UPDATE conversation_node SET kinds = array_remove(kinds, 'request') "
        "WHERE kinds && ARRAY['request']::text[]"
    )
    op.execute("DELETE FROM conversation_node WHERE cardinality(kinds) = 0")
    _replace(BEFORE)
