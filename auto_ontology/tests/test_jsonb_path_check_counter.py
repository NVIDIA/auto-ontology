# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent-level tests for JsonbPathCheckAgent's repair-attempt counter.

A fully self-applied fix (try_self_apply_fixes clears every mismatch) must
not spend a jsonb_path_repair_attempts slot, mirroring join_path_check.py's
equivalent early return. Only a mismatch that still needs reconstruction
after the self-apply pass should increment the counter.
"""

from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock

import pytest

from auto_ontology.retrieval.text_to_sql.agents import jsonb_path_check
from auto_ontology.retrieval.text_to_sql.agents.jsonb_path_check import (
    JsonbPathCheckAgent,
)


class _NullProbeExecutor:
    """Stands in for ProbeExecutor's context-manager protocol; the mismatch
    list itself comes from the monkeypatched find_jsonb_path_mismatches
    below, so this never needs to touch a real connector.
    """

    def __init__(self, connector: Any) -> None:
        pass

    def __enter__(self) -> "_NullProbeExecutor":
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        return False


def _make_state(path_state: Dict[str, Any]) -> Dict[str, Any]:
    return {"path_state": path_state, "connectors": []}


def _patch_common(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jsonb_path_check, "ProbeExecutor", _NullProbeExecutor)
    monkeypatch.setattr(
        jsonb_path_check, "resolve_connector_from_tables", lambda *a, **k: None
    )


def test_counter_preserved_for_two_self_applied_fixes_then_spent_on_third(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three calls in sequence, carrying the returned path_state forward
    each time: the first two fully self-apply (counter untouched, valid_sql),
    the third leaves a mismatch after the self-apply pass (counter spent,
    invalid_sql).
    """
    _patch_common(monkeypatch)

    # One mismatch list per call: the first two are always resolvable by
    # try_self_apply_fixes, the third survives it.
    mismatch = {
        "table": "aliens",
        "column": "data",
        "container": None,
        "used_key": "x",
        "available_keys": ["y"],
    }
    find_mismatches = MagicMock(side_effect=[[mismatch], [mismatch], [mismatch]])
    monkeypatch.setattr(jsonb_path_check, "find_jsonb_path_mismatches", find_mismatches)

    def fake_self_apply(
        mismatches: List[Dict[str, Any]], sql: str
    ) -> Tuple[str, List[Dict[str, Any]]]:
        call_index = find_mismatches.call_count
        if call_index <= 2:
            return "SELECT data->'flattened'->>'key' FROM aliens", []
        return sql, mismatches

    monkeypatch.setattr(jsonb_path_check, "try_self_apply_fixes", fake_self_apply)

    agent = JsonbPathCheckAgent()
    path_state: Dict[str, Any] = {
        "sql_code": "SELECT data->>'flattened.key' FROM aliens",
        "jsonb_path_repair_attempts": 0,
        "relevant_tables": [],
    }

    first = agent.execute(_make_state(path_state))
    assert first["decision"] == "valid_sql"
    assert first["path_state"]["jsonb_path_repair_attempts"] == 0
    path_state = first["path_state"]

    second = agent.execute(_make_state(path_state))
    assert second["decision"] == "valid_sql"
    assert second["path_state"]["jsonb_path_repair_attempts"] == 0
    path_state = second["path_state"]

    third = agent.execute(_make_state(path_state))
    assert third["decision"] == "invalid_sql"
    assert third["path_state"]["jsonb_path_repair_attempts"] == 1
