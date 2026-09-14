# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for find_case_dirty_join_mismatches (join_path_check.py) — a
compound JOIN where one predicate is the graph-verified relationship and
another, unverified predicate silently rejects real matches purely due to
casing/whitespace, confirmed by inspecting the rejected rows themselves
(not just a row-count difference, which can't tell dirty data apart from a
predicate correctly excluding a genuinely different row).

Scenario mirrors the real labor_certification_applications_13 bug: joining
worksite to case_worksite on the verified addr1 key plus an unverified
city/state pair rejects rows where wcity/wscity disagree only in
casing/whitespace for the same physical worksite.
"""

from unittest.mock import patch

import pandas as pd

from gsf.retrieval.text_to_sql.db_probe.executor import ProbeExecutor
from gsf.retrieval.text_to_sql.db_probe.join_path_check import (
    build_case_dirty_join_repair_error,
    find_case_dirty_join_mismatches,
)

_VERIFIED_HOP = [
    {
        "source_schema": "public",
        "source_table": "worksite",
        "source_column": "w_addr1",
        "target_schema": "public",
        "target_table": "case_worksite",
        "target_column": "ws_addr1",
    }
]


def _column_id(table_name, column_name, database_name=None):
    return f"{table_name}.{column_name}"


def _join_path_only_addr1_verified(col_a_id, col_b_id):
    if {col_a_id, col_b_id} == {"worksite.w_addr1", "case_worksite.ws_addr1"}:
        return _VERIFIED_HOP
    return []


def _join_path_nothing_verified(col_a_id, col_b_id):
    return []


class _ScriptedConnector:
    """Fake connector returning pre-scripted DataFrames in call order."""

    def __init__(self, results: list[pd.DataFrame]) -> None:
        self._results = list(results)

    def execute(self, sql: str) -> pd.DataFrame:
        return self._results.pop(0)


_JOIN_SQL = (
    "SELECT w.wstate FROM public.worksite w "
    "JOIN public.case_worksite cw "
    "ON w.w_addr1 = cw.ws_addr1 AND w.wstate = cw.wsstate"
)

_VERIFIED_PATCH = patch(
    "gsf.retrieval.text_to_sql.db_probe.join_path_check.find_join_path",
    side_effect=_join_path_only_addr1_verified,
)
_COL_ID_PATCH = patch(
    "gsf.retrieval.text_to_sql.db_probe.join_path_check.find_column_id_by_table_and_name",
    side_effect=_column_id,
)


@_VERIFIED_PATCH
@_COL_ID_PATCH
def test_flagged_when_every_rejected_row_is_casing_only(_mock_col_id, _mock_join_path):
    connector = _ScriptedConnector(
        [pd.DataFrame({"total_rejects": [13], "casing_only_rejects": [13]})]
    )
    executor = ProbeExecutor(connector=connector)

    mismatches = find_case_dirty_join_mismatches(
        executor, "postgres", _JOIN_SQL, "labor_certification_applications"
    )

    assert len(mismatches) == 1
    m = mismatches[0]
    assert m["verdict"] == "case_dirty_join"
    assert m["rejected_rows"] == 13
    assert m["verified_columns"] == ["worksite.w_addr1 = case_worksite.ws_addr1"]
    assert m["extra_columns"] == ["worksite.wstate = case_worksite.wsstate"]


@_VERIFIED_PATCH
@_COL_ID_PATCH
def test_not_flagged_when_some_rejected_rows_differ_for_real(
    _mock_col_id, _mock_join_path
):
    """Only 8 of 13 rejected rows agree once normalized — the other 5 are a
    genuine difference, so the extra predicate may be doing real
    disambiguation work. Must stay silent rather than guess.
    """
    connector = _ScriptedConnector(
        [pd.DataFrame({"total_rejects": [13], "casing_only_rejects": [8]})]
    )
    executor = ProbeExecutor(connector=connector)

    mismatches = find_case_dirty_join_mismatches(
        executor, "postgres", _JOIN_SQL, "labor_certification_applications"
    )

    assert mismatches == []


@_VERIFIED_PATCH
@_COL_ID_PATCH
def test_not_flagged_when_extra_predicate_rejects_nothing(
    _mock_col_id, _mock_join_path
):
    connector = _ScriptedConnector(
        [pd.DataFrame({"total_rejects": [0], "casing_only_rejects": [0]})]
    )
    executor = ProbeExecutor(connector=connector)

    mismatches = find_case_dirty_join_mismatches(
        executor, "postgres", _JOIN_SQL, "labor_certification_applications"
    )

    assert mismatches == []


@patch(
    "gsf.retrieval.text_to_sql.db_probe.join_path_check.find_join_path",
    side_effect=_join_path_nothing_verified,
)
@patch(
    "gsf.retrieval.text_to_sql.db_probe.join_path_check.find_column_id_by_table_and_name",
    side_effect=_column_id,
)
def test_not_flagged_when_no_predicate_is_graph_verified(_mock_col_id, _mock_join_path):
    """No known-good "core" predicate to test the other one against — the
    check must stay silent rather than guess which predicate (if either) is
    the real relationship. That's find_join_path_mismatches' job.
    """
    connector = _ScriptedConnector([])  # no probes should even be attempted
    executor = ProbeExecutor(connector=connector)

    mismatches = find_case_dirty_join_mismatches(
        executor, "postgres", _JOIN_SQL, "labor_certification_applications"
    )

    assert mismatches == []


def test_not_flagged_with_a_single_join_predicate():
    """Only one equality predicate between the two tables — nothing "extra"
    to be suspicious of.
    """
    sql = (
        "SELECT w.wstate FROM public.worksite w "
        "JOIN public.case_worksite cw ON w.w_addr1 = cw.ws_addr1"
    )
    executor = ProbeExecutor(connector=None)

    mismatches = find_case_dirty_join_mismatches(
        executor, "postgres", sql, "labor_certification_applications"
    )

    assert mismatches == []


def test_dead_executor_reports_nothing():
    """No connector available — the probe fails gracefully, no mismatch."""
    executor = ProbeExecutor(connector=None)

    with _VERIFIED_PATCH, _COL_ID_PATCH:
        mismatches = find_case_dirty_join_mismatches(
            executor, "postgres", _JOIN_SQL, "labor_certification_applications"
        )

    assert mismatches == []


def test_build_case_dirty_join_repair_error_renders_expected_content():
    mismatch = {
        "table_a": "worksite",
        "table_b": "case_worksite",
        "verified_columns": ["worksite.w_addr1 = case_worksite.ws_addr1"],
        "extra_columns": ["worksite.wstate = case_worksite.wsstate"],
        "rejected_rows": 13,
        "verdict": "case_dirty_join",
    }

    error = build_case_dirty_join_repair_error([mismatch])

    assert "13" in error
    assert "worksite.wstate = case_worksite.wsstate" in error
    assert "LOWER(TRIM(" in error
    assert "Do NOT drop the condition" in error
