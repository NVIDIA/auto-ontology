# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""widen conversation_node kinds with target and proposed

Two kinds that the original five could not express without lying about what a
turn said:

``target``
    a value wanted but not reached -- "get latency down to 290ms". Stored as
    ``state`` it reads as a measurement, and sits in the same revision chain as
    the real readings, so the gap between a goal and the figure looks like the
    figure having changed.
``proposed``
    something shown, tried or suggested but not adopted -- config a user pastes
    saying "not sure if this is the right approach". Stored as ``fact`` it
    reads as the configuration in force.

Also drops a stray duplicate. ``conversation_node_kinds_value`` exists on
databases where the check was widened by hand before this migration was
written: the schema declares that name, the metadata naming convention renders
it ``ck_conversation_node_conversation_node_kinds_value``, and a hand-written
``ALTER TABLE ... ADD CONSTRAINT`` under the declared name creates a second,
independently-enforced copy. The narrower of the two then rejects rows the
wider one allows, which is how this was found -- an extraction run died on
``proposed`` at window 80 of 100 against a database whose other copy of the
constraint permitted it.

Revision ID: d2b6f4a81e05
Revises: a1c7e3f20b94
Create Date: 2026-10-21 11:48:00.000000

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d2b6f4a81e05"
down_revision: Union[str, Sequence[str], None] = "a1c7e3f20b94"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

#: What ``schema.py`` calls it. ``op`` runs this through the metadata naming
#: convention, so creating under this name produces ``RENDERED`` in Postgres --
#: which is the point, and why the hand-applied copy is a duplicate rather than
#: a replacement.
DECLARED = "conversation_node_kinds_value"

#: What the convention makes of it, and what the database actually holds.
RENDERED = f"ck_conversation_node_{DECLARED}"

BEFORE = ("object", "fact", "preference", "event", "state")
AFTER = (*BEFORE, "target", "proposed")


def _check(kinds: Sequence[str]) -> str:
    values = ", ".join(f"'{kind}'" for kind in kinds)
    return f"kinds <@ ARRAY[{values}]::text[] AND array_length(kinds, 1) >= 1"


def _replace(kinds: Sequence[str]) -> None:
    """Leave exactly one check on ``kinds``, admitting *kinds*."""
    for name in (RENDERED, DECLARED):
        # IF EXISTS because only a hand-patched database carries DECLARED as a
        # constraint of its own; a clean one has only RENDERED.
        op.execute(f"ALTER TABLE conversation_node DROP CONSTRAINT IF EXISTS {name}")
    op.create_check_constraint(DECLARED, "conversation_node", _check(kinds))


def upgrade() -> None:
    """Upgrade schema."""
    _replace(AFTER)


def downgrade() -> None:
    """Downgrade schema.

    Rows using the two new kinds have to go first. They are not representable
    under the narrower check and there is no honest kind to move them to --
    calling a goal a ``state`` is the thing this migration exists to stop.
    """
    op.execute(
        "DELETE FROM conversation_node "
        "WHERE kinds && ARRAY['target', 'proposed']::text[]"
    )
    _replace(BEFORE)
