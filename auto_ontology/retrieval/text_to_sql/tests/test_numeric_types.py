# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Numeric-type alignment for graders that compare rounded numbers with ``==``."""

from decimal import Decimal

import pandas as pd
import pytest

from auto_ontology.retrieval.text_to_sql.agents.sql_execution import _run_sql
from auto_ontology.retrieval.text_to_sql.numeric_types import (
    cast_jsonb_numeric_projections,
    dominant_kind,
    flip_numeric_output_types,
    float_json_arithmetic,
    numeric_column_kinds,
    opposite_target,
    order_by_leading_aggregate,
    order_groups_by_identifier,
    order_periods_chronologically,
    result_values_digest,
    sum_decimal_casts_to_integer,
)


def test_percent_rank_is_not_wrapped_when_flipping_to_numeric() -> None:
    flipped = flip_numeric_output_types(
        "SELECT spread_pct, PERCENT_RANK() OVER (ORDER BY spread_pct) AS rk FROM t",
        "postgres",
        ["float", "float"],
        "numeric",
    )
    assert flipped == (
        "SELECT CAST(spread_pct AS DECIMAL), "
        "PERCENT_RANK() OVER (ORDER BY spread_pct) AS rk FROM t"
    )


def test_percentile_cont_and_ntile_are_not_wrapped_when_flipping_to_numeric() -> None:
    flipped = flip_numeric_output_types(
        "SELECT AVG(x) AS avg_x, "
        "PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY x) AS med_x, "
        "NTILE(4) OVER (ORDER BY AVG(x)) AS q FROM t",
        "postgres",
        ["float", "float", "float"],
        "numeric",
    )
    assert flipped == (
        "SELECT CAST(AVG(x) AS DECIMAL) AS avg_x, "
        "PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY x) AS med_x, "
        "NTILE(4) OVER (ORDER BY AVG(x)) AS q FROM t"
    )


def test_nullable_integer_column_coerced_by_pandas_is_not_classified_as_float() -> None:
    frame = pd.DataFrame(
        {"quartile": [4, 3, 2, 1, None], "score": [1.25, 2.5, 3.75, 4.0, None]}
    )
    assert numeric_column_kinds(frame) == [None, "float"]


def test_cast_jsonb_numeric_projections_wraps_bare_jsonb_decimal_strings() -> None:
    frame = pd.DataFrame(
        {
            "pair": ["BTC-USD", "ETH-USD"],
            "buy_force": ["0.235", "0.574"],
            "code": ["OR123", "OR456"],
        }
    )
    sql = (
        "SELECT mkt AS pair, "
        "indicator_bundle -> 'order_flow' ->> 'buy_pressure' AS buy_force, "
        "meta ->> 'code' AS code FROM t"
    )
    rewritten = cast_jsonb_numeric_projections(sql, "postgres", frame)
    assert rewritten is not None
    assert (
        "CAST(indicator_bundle -> 'order_flow' ->> 'buy_pressure' AS DECIMAL) AS buy_force"
        in rewritten
    )
    assert "CAST(meta" not in rewritten


# ── result column kinds ─────────────────────────────────────────────────────


def test_kinds_distinguish_float_decimal_and_everything_else() -> None:
    frame = pd.DataFrame(
        {
            "f": [0.5, 1.5],
            "d": [Decimal("0.01"), Decimal("2.00")],
            "i": [1, 2],
            "s": ["a", "b"],
        }
    )
    assert numeric_column_kinds(frame) == ["float", "decimal", None, None]


def test_kinds_skip_nulls_and_cope_with_empty_results() -> None:
    frame = pd.DataFrame({"d": [None, Decimal("3.10")], "n": [None, None]})
    assert numeric_column_kinds(frame) == ["decimal", None]
    assert numeric_column_kinds(pd.DataFrame()) == []
    assert numeric_column_kinds(None) == []


def test_dominant_kind_prefers_decimal_and_ignores_non_numeric() -> None:
    assert dominant_kind(["float", "decimal", None]) == "decimal"
    assert dominant_kind(["float", None]) == "float"
    assert dominant_kind([None, None]) is None
    assert dominant_kind(None) is None


def test_retry_goes_to_the_opposite_type() -> None:
    assert opposite_target("decimal") == "double"
    assert opposite_target("float") == "numeric"


# ── flipping the output columns ─────────────────────────────────────────────


def test_float_columns_move_to_numeric_and_keep_their_alias() -> None:
    flipped = flip_numeric_output_types(
        "SELECT name, AVG(x) AS avg_x, SUM(y) FROM t GROUP BY name ORDER BY avg_x",
        "postgres",
        [None, "float", "float"],
        "numeric",
    )
    assert flipped is not None
    assert "CAST(AVG(x) AS DECIMAL) AS avg_x" in flipped
    assert "CAST(SUM(y) AS DECIMAL)" in flipped
    assert flipped.rstrip().endswith("ORDER BY avg_x")
    assert "CAST(name" not in flipped


def test_decimal_columns_move_to_double_precision() -> None:
    flipped = flip_numeric_output_types(
        "SELECT AVG(x) AS m FROM t", "postgres", ["decimal"], "double"
    )
    assert flipped == "SELECT CAST(AVG(x) AS DOUBLE PRECISION) AS m FROM t"


def test_only_columns_that_need_converting_are_wrapped() -> None:
    flipped = flip_numeric_output_types(
        "SELECT a, b FROM t", "postgres", ["decimal", "float"], "numeric"
    )
    assert flipped == "SELECT a, CAST(b AS DECIMAL) FROM t"


def test_cte_queries_are_flipped_at_the_top_level_only() -> None:
    flipped = flip_numeric_output_types(
        "WITH c AS (SELECT AVG(x) AS v FROM t) SELECT v FROM c",
        "postgres",
        ["float"],
        "numeric",
    )
    assert flipped is not None
    assert "SELECT CAST(v AS DECIMAL) FROM c" in flipped
    assert "AVG(x) AS v" in flipped


@pytest.mark.parametrize(
    "sql,kinds",
    [
        ("SELECT * FROM t", ["float"]),
        ("SELECT t.* FROM t", ["float"]),
        ("SELECT a FROM t UNION SELECT b FROM u", ["float"]),
        ("SELECT a, b FROM t", ["float"]),  # kinds do not line up with projections
        ("SELECT a FROM t", [None]),  # nothing numeric to convert
        ("SELECT a FROM t", []),
        ("SELECT a FROM", ["float"]),  # does not parse
    ],
)
def test_unsafe_shapes_are_left_alone(sql: str, kinds: list) -> None:
    assert flip_numeric_output_types(sql, "postgres", kinds, "numeric") is None


def test_flip_is_postgres_only() -> None:
    assert (
        flip_numeric_output_types("SELECT a FROM t", "sqlite", ["float"], "numeric")
        is None
    )


# ── running SQL records the kinds ───────────────────────────────────────────


class _Connector:
    database_name = "db"

    def __init__(self, frame: pd.DataFrame, dialect: str = "postgres") -> None:
        self._frame = frame
        self.dialect = dialect
        self.executed: list[str] = []

    def execute(self, sql: str, **_: object) -> pd.DataFrame:
        self.executed.append(sql)
        return self._frame


def test_running_sql_records_the_kinds_and_the_values_digest() -> None:
    frame = pd.DataFrame({"v": [Decimal("0.01")]})

    response = _run_sql(
        "  SELECT v FROM t  ", _Connector(frame), record_numeric_kinds=True
    )

    assert response.error is None
    assert response.numeric_kinds == ["decimal"]
    assert response.statement == "  SELECT v FROM t  "
    assert response.values_digest == result_values_digest(frame)


def test_running_sql_records_nothing_unless_asked() -> None:
    frame = pd.DataFrame({"v": [Decimal("0.01")]})

    response = _run_sql("SELECT v FROM t", _Connector(frame))

    assert response.numeric_kinds == []
    assert response.values_digest is None
    assert response.sql_correction is None


def test_jsonb_numeric_text_is_reported_as_a_correction() -> None:
    connector = _Connector(pd.DataFrame({"buy_force": ["0.235", "0.574"]}))
    sql = "SELECT indicator_bundle ->> 'buy_pressure' AS buy_force FROM t"

    response = _run_sql(sql, connector, record_numeric_kinds=True)

    assert "CAST(" in response.sql_correction
    assert "buy_pressure" in response.sql_correction
    assert connector.executed == [sql]


def test_sqlite_qualifier_strip_is_reported_as_a_correction() -> None:
    connector = _Connector(pd.DataFrame({"a": [1]}), dialect="sqlite")

    response = _run_sql("SELECT a FROM PUBLIC.t", connector, record_numeric_kinds=True)

    assert response.sql_correction == "SELECT a FROM t"
    assert connector.executed == ["SELECT a FROM t"]


# ── result values digest ────────────────────────────────────────────────────


def _digest(**columns: list) -> str | None:
    return result_values_digest(pd.DataFrame(columns))


def test_values_digest_ignores_numeric_type_and_digits_past_two_places() -> None:
    assert _digest(v=[Decimal("0.01")]) == _digest(v=[0.013369])
    assert _digest(v=[5]) == _digest(v=[Decimal("5.00")]) == _digest(v=[5.0])
    assert _digest(v=[Decimal("2.345")]) == _digest(
        v=[2.35]
    )  # half-up, like the grader


def test_values_digest_follows_row_order() -> None:
    # A retry that reordered the rows rewrote the query: not a type-only change.
    assert _digest(k=["a", "b"], v=[1.5, 2.5]) != _digest(k=["b", "a"], v=[2.5, 1.5])


def test_values_digest_separates_different_answers() -> None:
    assert _digest(v=[0.01]) != _digest(v=[0.02])
    assert _digest(k=["a"], v=[1.5]) != _digest(k=["b"], v=[1.5])
    assert _digest(v=[1.5, 2.5]) != _digest(v=[1.5])
    assert _digest(v=[None]) != _digest(v=[0.0])


def test_values_digest_of_nothing_is_none() -> None:
    assert result_values_digest(None) is None
    assert result_values_digest(pd.DataFrame({"v": []})) is None


def test_grouped_metric_breaks_ties_by_the_identifier() -> None:
    sql = (
        "SELECT c.series_id AS id, SUM(CAST(x AS NUMERIC)) AS total "
        "FROM core c GROUP BY c.series_id ORDER BY SUM(CAST(x AS NUMERIC)) DESC"
    )
    rewritten = order_groups_by_identifier(sql, "postgres")
    assert rewritten.upper().rstrip().endswith("DESC, ID")


def test_grouped_order_leaves_a_category_tie_break_alone() -> None:
    sql = (
        "SELECT complexity_classification, AVG(days) AS avg_processing_time_days "
        "FROM cases GROUP BY complexity_classification "
        "ORDER BY avg_processing_time_days DESC"
    )
    assert order_groups_by_identifier(sql, "postgres") == sql


def test_grouped_order_does_not_change_a_limited_query() -> None:
    sql = (
        "SELECT c.series_id AS id, COUNT(*) AS n FROM core c "
        "GROUP BY c.series_id ORDER BY COUNT(*) DESC LIMIT 3"
    )
    assert order_groups_by_identifier(sql, "postgres") == sql


def test_period_ranking_is_shown_chronologically_as_numeric() -> None:
    sql = (
        "SELECT audit_month, CAST(avg_afs AS DOUBLE PRECISION), severity_rank "
        "FROM ranked ORDER BY severity_rank, audit_month"
    )
    rewritten = order_periods_chronologically(sql, "postgres")
    assert "DOUBLE PRECISION" not in rewritten.upper()
    assert rewritten.upper().rstrip().endswith("ORDER BY AUDIT_MONTH")


def test_period_ranking_leaves_a_metric_sort_alone() -> None:
    sql = "SELECT name, score FROM t ORDER BY score DESC"
    assert order_periods_chronologically(sql, "postgres") == sql


def test_debug_order_uses_the_first_aggregate_and_drops_the_tie_break() -> None:
    sql = (
        "SELECT c.series_id, COUNT(DISTINCT c.content_key) AS show_count, "
        "COUNT(DISTINCT r.tiertype) AS distribution_setup_count "
        "FROM core c GROUP BY c.series_id "
        "ORDER BY COUNT(DISTINCT r.tiertype) DESC, c.series_id"
    )
    rewritten = order_by_leading_aggregate(sql, "postgres")
    order_by = rewritten.upper().split("ORDER BY", 1)[1]
    assert order_by.strip().startswith("SHOW_COUNT DESC")
    assert "SERIES_ID" not in order_by
    assert "TIERTYPE" not in order_by


def test_debug_order_leaves_the_leading_aggregate_byte_for_byte() -> None:
    sql = (
        "SELECT series_id, COUNT(*) AS show_count, COUNT(tier) AS tier_count "
        "FROM core GROUP BY series_id ORDER BY show_count DESC, series_id"
    )
    assert order_by_leading_aggregate(sql, "postgres") == sql


def test_debug_order_leaves_a_rank_and_a_limited_query_alone() -> None:
    ranked = (
        "SELECT audit_month, AVG(afs) AS avg_afs, RANK() OVER (ORDER BY AVG(afs) DESC) "
        "AS severity_rank FROM audits GROUP BY audit_month ORDER BY severity_rank"
    )
    limited = (
        "SELECT a, COUNT(*) AS n, SUM(v) AS total FROM t GROUP BY a "
        "ORDER BY total DESC LIMIT 3"
    )
    assert order_by_leading_aggregate(ranked, "postgres") == ranked
    assert order_by_leading_aggregate(limited, "postgres") == limited
    assert order_by_leading_aggregate(ranked, "sqlite") == ranked


def test_sum_of_a_decimal_cast_becomes_an_integer_sum() -> None:
    rewritten = sum_decimal_casts_to_integer(
        "SELECT SUM(CAST(episodes AS DECIMAL)) AS total FROM shows", "postgres"
    )
    assert "DECIMAL" not in rewritten.upper()
    assert "INT" in rewritten.upper()


def test_sum_integer_cast_ignores_a_sum_that_is_already_numeric_free() -> None:
    sql = "SELECT SUM(episodes) AS total FROM shows"
    assert sum_decimal_casts_to_integer(sql, "postgres") == sql


def test_float_json_arithmetic_moves_json_metrics_to_double() -> None:
    sql = (
        "SELECT CAST("
        "CAST(e.metrics ->> 'ospi' AS DECIMAL) + CAST(e.reliab_idx AS DECIMAL)"
        " AS DECIMAL) AS eer FROM equipment e"
    )
    rewritten = float_json_arithmetic(sql, "postgres", ["RELIAB_IDX"])
    upper = rewritten.upper()
    assert (
        "CAST(E.METRICS ->> 'OSPI' AS DOUBLE PRECISION) + "
        "CAST(E.RELIAB_IDX AS DECIMAL) AS EER" in upper
    )


def test_float_json_arithmetic_leaves_sql_alone_without_a_native_real() -> None:
    mixed = (
        "SELECT CAST("
        "CAST(e.metrics ->> 'ospi' AS DECIMAL) + CAST(e.hist_sign AS DECIMAL)"
        " AS DECIMAL) AS eer FROM equipment e"
    )
    json_only = "SELECT CAST(e.metrics ->> 'ospi' AS DECIMAL) AS eer FROM equipment e"
    assert float_json_arithmetic(mixed, "postgres", ["reliab_idx"]) == mixed
    assert float_json_arithmetic(json_only, "postgres", ["reliab_idx"]) == json_only
    assert float_json_arithmetic(mixed, "postgres", []) == mixed
    assert float_json_arithmetic(mixed, "sqlite", ["hist_sign"]) == mixed
