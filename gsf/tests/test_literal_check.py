# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the case_duplicate detection in find_literal_mismatches
(literal_check.py) — a filter literal that exactly matches a real value, but
that same real-world value also appears under other casing/whitespace
elsewhere in the column, so an exact-match filter silently misses those rows.
"""

import pandas as pd

from gsf.retrieval.text_to_sql.db_probe.executor import ProbeExecutor
from gsf.retrieval.text_to_sql.db_probe.literal_check import (
    build_value_repair_error,
    find_literal_mismatches,
)


class _ScriptedConnector:
    """Fake connector returning pre-scripted DataFrames in call order."""

    def __init__(self, results: list[pd.DataFrame]) -> None:
        self._results = list(results)

    def execute(self, sql: str) -> pd.DataFrame:
        return self._results.pop(0)


def _dead_executor() -> ProbeExecutor:
    """A ProbeExecutor with no connector — every .run() fails gracefully
    (ok=False), simulating "the live probe couldn't confirm anything" without
    needing a real database.
    """
    return ProbeExecutor(connector=None)


def test_case_duplicate_flagged_when_literal_exact_matches_but_has_siblings():
    """statustag = 'Certified' exactly matches a real value, but the same
    real-world value also exists as 'CERTIFIED'/'certified' in the column —
    an exact-match filter silently misses those rows. (Mirrors the real
    labor_certification_applications.cases.statustag data quality issue.)
    """
    connector = _ScriptedConnector(
        [pd.DataFrame({"value": ["Certified", "CERTIFIED", "certified", "Denied"]})]
    )
    executor = ProbeExecutor(connector=connector)
    sql = "SELECT * FROM cases WHERE statustag = 'Certified'"

    mismatches = find_literal_mismatches(executor, "postgres", sql)

    assert len(mismatches) == 1
    m = mismatches[0]
    assert m["kind"] == "case_duplicate"
    assert m["table"] == "cases"
    assert m["column"] == "statustag"
    assert m["used"] == "Certified"
    assert set(m["siblings"]) == {"Certified", "CERTIFIED", "certified"}


def test_no_mismatch_when_exact_match_has_no_case_siblings():
    """The clean-data case (e.g. an enum column like visacls): an exact
    match with no case-variant siblings must not be flagged.
    """
    connector = _ScriptedConnector([pd.DataFrame({"value": ["H-1B", "H-1B1 Chile"]})])
    executor = ProbeExecutor(connector=connector)
    sql = "SELECT * FROM cases WHERE visacls = 'H-1B'"

    mismatches = find_literal_mismatches(executor, "postgres", sql)

    assert mismatches == []


def test_near_miss_reports_every_case_sibling_as_a_suggestion():
    """A literal with no exact match, but several case variants of the same
    normalized value present, should surface every sibling as a suggestion —
    not just the first one encountered (this was a latent bug in the old
    single-valued actual_by_norm map, fixed as part of adding case_duplicate
    detection since both share the same normalized-value grouping).
    """
    connector = _ScriptedConnector(
        [pd.DataFrame({"value": ["CERTIFIED", "certified", "Denied"]})]
    )
    executor = ProbeExecutor(connector=connector)
    sql = "SELECT * FROM cases WHERE statustag = 'Certified'"

    mismatches = find_literal_mismatches(executor, "postgres", sql)

    assert len(mismatches) == 1
    m = mismatches[0]
    assert m["kind"] == "string"
    assert set(m["suggested"]) == {"CERTIFIED", "certified"}


def test_dead_executor_reports_nothing():
    """No connector available — probe fails gracefully, no mismatch."""
    sql = "SELECT * FROM cases WHERE statustag = 'Certified'"
    mismatches = find_literal_mismatches(_dead_executor(), "postgres", sql)
    assert mismatches == []


def test_build_value_repair_error_renders_case_duplicate_section():
    mismatch = {
        "kind": "case_duplicate",
        "table": "cases",
        "column": "statustag",
        "used": "Certified",
        "siblings": ["CERTIFIED", "Certified", "certified"],
    }

    error = build_value_repair_error([mismatch])

    assert "LOWER(TRIM(" in error
    assert "'CERTIFIED'" in error
    assert "'certified'" in error
    # The literal's own exact-match variant shouldn't be listed back as a
    # "different" sibling of itself.
    assert "'Certified', 'CERTIFIED'" not in error


def test_case_duplicate_mismatch_casts_to_text_on_postgres():
    """A Postgres ENUM column has no LOWER()/TRIM() overload without an
    explicit ::text cast (observed directly: `function pg_catalog.btrim
    (enum_visa_class) does not exist`) — the suggested rewrite must include
    it when the dialect is Postgres.
    """
    connector = _ScriptedConnector(
        [pd.DataFrame({"value": ["Certified", "CERTIFIED", "certified", "Denied"]})]
    )
    executor = ProbeExecutor(connector=connector)
    sql = "SELECT * FROM cases WHERE statustag = 'Certified'"

    mismatches = find_literal_mismatches(executor, "postgres", sql)

    assert mismatches[0]["dialect"] == "postgres"
    error = build_value_repair_error(mismatches)
    assert "statustag::text" in error


def test_case_duplicate_mismatch_has_no_cast_on_unrecognized_dialect():
    """No column-type information is available here, so an unrecognized (or
    absent) dialect must never guess at cast syntax — leave the column bare.
    """
    connector = _ScriptedConnector(
        [pd.DataFrame({"value": ["Certified", "CERTIFIED", "certified", "Denied"]})]
    )
    executor = ProbeExecutor(connector=connector)
    sql = "SELECT * FROM cases WHERE statustag = 'Certified'"

    mismatches = find_literal_mismatches(executor, None, sql)

    assert mismatches[0]["dialect"] is None
    error = build_value_repair_error(mismatches)
    assert "::text" not in error
    assert "LOWER(TRIM(statustag))" in error


def test_case_duplicate_like_flagged_when_pattern_misses_case_variants():
    """statustag LIKE 'Certified%' is case-sensitive, so it matches 'Certified'
    but misses the real 'CERTIFIED'/'certified' rows — the same underlying
    labor_certification_applications.cases.statustag issue, just expressed as
    a LIKE pattern instead of `=`.
    """
    connector = _ScriptedConnector(
        [pd.DataFrame({"value": ["Certified", "CERTIFIED", "certified", "Denied"]})]
    )
    executor = ProbeExecutor(connector=connector)
    sql = "SELECT * FROM cases WHERE statustag LIKE 'Certified%'"

    mismatches = find_literal_mismatches(executor, "postgres", sql)

    assert len(mismatches) == 1
    m = mismatches[0]
    assert m["kind"] == "case_duplicate_like"
    assert m["table"] == "cases"
    assert m["column"] == "statustag"
    assert m["pattern"] == "Certified%"
    assert set(m["missed"]) == {"CERTIFIED", "certified"}


def test_no_mismatch_when_like_pattern_already_matches_every_variant():
    connector = _ScriptedConnector(
        [pd.DataFrame({"value": ["Certified", "CERTIFIED", "certified", "Denied"]})]
    )
    executor = ProbeExecutor(connector=connector)
    sql = "SELECT * FROM cases WHERE statustag ILIKE 'Certified%'"

    mismatches = find_literal_mismatches(executor, "postgres", sql)

    assert mismatches == []


def test_no_mismatch_when_like_pattern_has_no_case_variants_to_miss():
    connector = _ScriptedConnector([pd.DataFrame({"value": ["H-1B", "H-1B1 Chile"]})])
    executor = ProbeExecutor(connector=connector)
    sql = "SELECT * FROM cases WHERE visacls LIKE 'H-1B%'"

    mismatches = find_literal_mismatches(executor, "postgres", sql)

    assert mismatches == []


def test_not_like_is_ignored():
    """NOT LIKE is a negation, out of scope — same rule as `=`/IN."""
    connector = _ScriptedConnector(
        [pd.DataFrame({"value": ["Certified", "CERTIFIED", "certified", "Denied"]})]
    )
    executor = ProbeExecutor(connector=connector)
    sql = "SELECT * FROM cases WHERE statustag NOT LIKE 'Certified%'"

    mismatches = find_literal_mismatches(executor, "postgres", sql)

    assert mismatches == []


def test_build_value_repair_error_renders_case_duplicate_like_section():
    mismatch = {
        "kind": "case_duplicate_like",
        "table": "cases",
        "column": "statustag",
        "pattern": "Certified%",
        "missed": ["CERTIFIED", "certified"],
    }

    error = build_value_repair_error([mismatch])

    assert "ILIKE" in error
    assert "'CERTIFIED'" in error
    assert "'certified'" in error


def test_build_value_repair_error_keeps_case_duplicate_separate_from_string_section():
    """The three mismatch kinds (string / case_duplicate / numeric_scale)
    must render into independent sections so reconstruction gets a targeted
    instruction per failure mode, not a merged, confusing one.
    """
    string_mismatch = {
        "kind": "string",
        "table": "cases",
        "column": "visacls",
        "used": "H1B",
        "suggested": ["H-1B"],
        "actual": ["H-1B", "H-1B1 Chile"],
    }
    case_duplicate_mismatch = {
        "kind": "case_duplicate",
        "table": "cases",
        "column": "statustag",
        "used": "Certified",
        "siblings": ["CERTIFIED", "Certified", "certified"],
    }

    error = build_value_repair_error([string_mismatch, case_duplicate_mismatch])

    assert "does not exist in the database" in error  # string section
    assert "silently misses those rows" in error  # case_duplicate section
