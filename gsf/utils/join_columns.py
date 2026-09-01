# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for the JOIN edge's ``join_columns`` property.

Neo4j node/relationship properties may only be primitives or arrays of
primitives — never a list of maps — so ``join_columns`` (a list of
``{"source": ..., "target": ...}`` column-name pairs) is stored as a JSON
string, the same trick used for ``Column.sample_values``
(see ``gsf.utils.sample_values``).
"""

from __future__ import annotations

import json
from typing import Any


def dump_join_columns(join_columns: list[dict[str, str]]) -> str:
    """Serialize a join's column pairs for storage as a Neo4j property."""
    return json.dumps(join_columns)


def parse_join_columns(raw: Any) -> list[dict[str, str]]:
    """Normalize a stored ``join_columns`` value (JSON string or list) back to a list."""
    if not raw:
        return []
    if isinstance(raw, list):
        return raw
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []
