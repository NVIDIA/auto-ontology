# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for closing "which stored value means X" gaps from the data."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from auto_ontology.retrieval.interactive import value_gaps
from auto_ontology.retrieval.interactive.value_gaps import (
    _distinct_values_sql,
    is_value_gap,
    match_value,
    record_value_notes,
    resolve_value_gaps,
)

_PENDING_GAP = (
    "pending match",
    'Match Status column resolved, but the exact value identifying a "pending" '
    "match status is still unspecified",
)
_STATUS_VALUES = ["Failed", "Matched", "In Progress", "Completed", "Pending"]


def _session(hits, connectors=None):
    return SimpleNamespace(
        task_id="t1",
        db_name="db",
        connectors=connectors if connectors is not None else [object()],
        _cached_resolved_hits=hits,
        _value_resolution_notes=[],
    )


def _contexts(*_args, **_kwargs):
    return {
        "a1": {
            "schema_name": "public",
            "table_name": "transplant_matching",
            "col_name": "match_status",
        }
    }


def _rows(values):
    return {"ok": True, "rows": [{"value": v} for v in values]}


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv("DB_PROBE_VALUE_GAP_LOOKUP", "1")


def _patch_run(values):
    return patch.object(value_gaps.ProbeExecutor, "run", return_value=_rows(values))


def test_is_value_gap_reads_the_missing_text() -> None:
    assert is_value_gap("the exact value/operator for a 'pending' status")
    assert is_value_gap("which label represents pancreas")
    assert not is_value_gap("the aggregation level is unspecified")
    assert not is_value_gap("")


def test_match_value_requires_every_word_of_the_value_in_the_term() -> None:
    assert match_value(_STATUS_VALUES, "pending match") == "Pending"
    assert match_value(_STATUS_VALUES, "matches in progress") == "In Progress"
    assert match_value(_STATUS_VALUES, "waiting list") is None


def test_match_value_ignores_bare_codes_and_numbers() -> None:
    assert match_value(["A", "B", "1", "2"], "grade a for 1 patient") is None


def test_match_value_is_ambiguous_for_two_unrelated_values() -> None:
    assert match_value(["Kidney", "Liver"], "kidney or liver list") is None


def test_match_value_prefers_the_more_specific_value() -> None:
    assert match_value(["Progress", "In Progress"], "in progress") == "In Progress"


def test_distinct_sql_quotes_schema_table_and_column() -> None:
    sql = _distinct_values_sql(
        {"schema_name": "public", "table_name": "T", "col_name": "Status"},
        "postgres",
        20,
    )
    assert sql == (
        'SELECT DISTINCT "T"."Status" AS value FROM "public"."T" '
        'WHERE "T"."Status" IS NOT NULL LIMIT 21'
    )
    assert _distinct_values_sql({"table_name": "", "col_name": "c"}, None, 5) is None


def test_disabled_by_default_leaves_gaps_untouched(monkeypatch) -> None:
    monkeypatch.delenv("DB_PROBE_VALUE_GAP_LOOKUP", raising=False)
    session = _session([("pending match", "txt", 0.1, "a1")])
    with _patch_run(_STATUS_VALUES) as run:
        remaining, notes = resolve_value_gaps(
            session, [_PENDING_GAP], fetch_contexts=_contexts
        )
    assert remaining == [_PENDING_GAP]
    assert notes == []
    run.assert_not_called()


def test_closes_a_value_gap_from_the_columns_distinct_values(enabled) -> None:
    other = ("rank order", "the sort direction is unspecified")
    session = _session([("pending match", "txt", 0.1, "a1")])
    with _patch_run(_STATUS_VALUES):
        remaining, notes = resolve_value_gaps(
            session, [other, _PENDING_GAP], fetch_contexts=_contexts
        )
    assert remaining == [other]
    assert len(notes) == 1
    assert "transplant_matching.match_status = 'Pending'" in notes[0]
    assert "pending match" in notes[0]


def test_non_value_gaps_never_trigger_a_probe(enabled) -> None:
    gap = ("pending match", "the aggregation semantics are unspecified")
    session = _session([("pending match", "txt", 0.1, "a1")])
    with _patch_run(_STATUS_VALUES) as run:
        remaining, notes = resolve_value_gaps(session, [gap], fetch_contexts=_contexts)
    assert remaining == [gap] and notes == []
    run.assert_not_called()


def test_gap_stays_open_when_the_term_matches_no_value(enabled) -> None:
    session = _session([("pending match", "txt", 0.1, "a1")])
    with _patch_run(["Failed", "Matched", "Completed"]):
        remaining, notes = resolve_value_gaps(
            session, [_PENDING_GAP], fetch_contexts=_contexts
        )
    assert remaining == [_PENDING_GAP] and notes == []


def test_gap_stays_open_when_the_column_has_too_many_values(enabled) -> None:
    many = [f"v{i}" for i in range(value_gaps.DB_PROBE_LOW_CARD_THRESHOLD + 1)]
    session = _session([("pending match", "txt", 0.1, "a1")])
    with _patch_run(many + ["Pending"]):
        remaining, notes = resolve_value_gaps(
            session, [_PENDING_GAP], fetch_contexts=_contexts
        )
    assert remaining == [_PENDING_GAP] and notes == []


def test_gap_stays_open_without_a_matching_resolved_term(enabled) -> None:
    session = _session([("urgency status", "txt", 0.1, "a1")])
    with _patch_run(_STATUS_VALUES) as run:
        remaining, notes = resolve_value_gaps(
            session, [_PENDING_GAP], fetch_contexts=_contexts
        )
    assert remaining == [_PENDING_GAP] and notes == []
    run.assert_not_called()


def test_gap_stays_open_without_a_single_connector(enabled) -> None:
    session = _session([("pending match", "txt", 0.1, "a1")], connectors=[])
    remaining, notes = resolve_value_gaps(
        session, [_PENDING_GAP], fetch_contexts=_contexts
    )
    assert remaining == [_PENDING_GAP] and notes == []


def test_context_failure_leaves_gaps_untouched(enabled) -> None:
    def boom(*_a, **_k):
        raise RuntimeError("catalog down")

    session = _session([("pending match", "txt", 0.1, "a1")])
    remaining, notes = resolve_value_gaps(session, [_PENDING_GAP], fetch_contexts=boom)
    assert remaining == [_PENDING_GAP] and notes == []


def test_failed_probe_leaves_gap_open(enabled) -> None:
    session = _session([("pending match", "txt", 0.1, "a1")])
    with patch.object(
        value_gaps.ProbeExecutor, "run", return_value={"ok": False, "rows": None}
    ):
        remaining, notes = resolve_value_gaps(
            session, [_PENDING_GAP], fetch_contexts=_contexts
        )
    assert remaining == [_PENDING_GAP] and notes == []


def test_record_value_notes_does_not_repeat_a_note() -> None:
    session = _session([])
    record_value_notes(session, ["a", "b"])
    record_value_notes(session, ["b", "c"])
    assert session._value_resolution_notes == ["a", "b", "c"]
