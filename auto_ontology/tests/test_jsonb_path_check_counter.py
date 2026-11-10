# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent-level tests for JsonbPathCheckAgent's repair-attempt counter.

A fully self-applied fix (try_self_apply_fixes clears every mismatch) must
not spend a jsonb_path_repair_attempts slot, mirroring join_path_check.py's
equivalent early return. Only a mismatch that still needs reconstruction
after the self-apply pass should increment the counter.
"""

from auto_ontology.retrieval.text_to_sql.agents import jsonb_path_check
from auto_ontology.retrieval.text_to_sql.agents.jsonb_path_check import (
    JsonbPathCheckAgent,
)


class _NullProbeExecutor:
    """Stands in for ProbeExecutor's context-manager protocol; the mismatch
    list itself comes from the monkeypatched find_jsonb_path_mismatches
    below, so this never needs to touch a real connector.
    """

    def __init__(self, connector):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def _make_state(attempts: int) -> dict:
    return {
        "path_state": {
            "sql_code": "SELECT data->>'flattened.key' FROM aliens",
            "jsonb_path_repair_attempts": attempts,
            "relevant_tables": [],
        },
        "connectors": [],
    }


def _patch_common(monkeypatch):
    monkeypatch.setattr(jsonb_path_check, "ProbeExecutor", _NullProbeExecutor)
    monkeypatch.setattr(
        jsonb_path_check, "resolve_connector_from_tables", lambda *a, **k: None
    )


def test_two_fully_self_applied_fixes_preserve_counter(monkeypatch):
    """Two calls in a row where try_self_apply_fixes resolves every
    mismatch must each return valid_sql without touching the counter.
    """
    _patch_common(monkeypatch)
    monkeypatch.setattr(
        jsonb_path_check,
        "find_jsonb_path_mismatches",
        lambda *a, **k: [{"table": "aliens", "column": "data", "used_key": "x"}],
    )
    monkeypatch.setattr(
        jsonb_path_check,
        "try_self_apply_fixes",
        lambda mismatches, sql: ("SELECT data->'flattened'->>'key' FROM aliens", []),
    )

    agent = JsonbPathCheckAgent()

    state = _make_state(attempts=0)
    result = agent.execute(state)
    assert result["decision"] == "valid_sql"
    assert result["path_state"]["jsonb_path_repair_attempts"] == 0

    state = _make_state(attempts=0)
    result = agent.execute(state)
    assert result["decision"] == "valid_sql"
    assert result["path_state"]["jsonb_path_repair_attempts"] == 0


def test_remaining_mismatch_after_self_apply_increments_and_fails(monkeypatch):
    """When a mismatch survives the self-apply pass, this call does spend
    a repair attempt and routes to reconstruction (invalid_sql) — the
    counter and decision a third, non-self-fixable call should produce
    after the two free passes above.
    """
    _patch_common(monkeypatch)
    monkeypatch.setattr(
        jsonb_path_check,
        "find_jsonb_path_mismatches",
        lambda *a, **k: [
            {
                "table": "aliens",
                "column": "data",
                "container": None,
                "used_key": "x",
                "available_keys": ["y"],
            }
        ],
    )
    monkeypatch.setattr(
        jsonb_path_check,
        "try_self_apply_fixes",
        lambda mismatches, sql: (sql, mismatches),
    )

    agent = JsonbPathCheckAgent()
    state = _make_state(attempts=0)
    result = agent.execute(state)

    assert result["decision"] == "invalid_sql"
    assert result["path_state"]["jsonb_path_repair_attempts"] == 1
