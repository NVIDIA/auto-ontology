# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Helm chart's pinned Alembic revision must match the head migration.

The backend's ``wait-for-catalog-schema`` init container blocks until
``public.alembic_version`` equals ``backend.alembicRevision``. That makes the value
load-bearing in a way a comment cannot enforce: pin it to a revision that never
arrives and every backend Pod waits forever; leave it behind after adding a
migration and the gate passes before the new schema is applied, which is the
bug it exists to prevent.
"""

from __future__ import annotations

import pathlib
import re

import pytest
from alembic.script import ScriptDirectory

ROOT = pathlib.Path(__file__).resolve().parents[3]
MIGRATIONS = ROOT / "alembic"
VALUES = ROOT / "helm" / "auto-ontology" / "values.yaml"


def _head_revisions() -> list[str]:
    """Use the same revision graph as ``alembic upgrade head``, including merges."""
    return ScriptDirectory(str(MIGRATIONS)).get_heads()


def test_there_is_exactly_one_head_migration() -> None:
    """Two heads mean `upgrade head` is ambiguous and the gate cannot be pinned."""
    heads = _head_revisions()
    assert len(heads) == 1, f"expected a single head, found {heads}"


@pytest.mark.skipif(not VALUES.exists(), reason="helm chart not present")
def test_the_chart_pins_the_head_revision() -> None:
    head = _head_revisions()[0]
    match = re.search(r"^\s*alembicRevision:\s*(\S+)", VALUES.read_text(), re.M)
    assert match, "backend.alembicRevision is missing from values.yaml"
    assert match.group(1) == head, (
        f"values.yaml pins {match.group(1)} but the head migration is {head}. "
        "Bump it in the same commit as the migration, or every backend Pod "
        "will block on a revision that never arrives."
    )
