"""Tests for process_table taxonomy compilation."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

import pytest

from gsf.semantic.visit_enter import (
    _MAX_CONSECUTIVE_SAMPLE_FAILURES,
    _distinct_values_if_low_cardinality,
    _PROFILING_QUERY_TIMEOUT_SECONDS,
    _json_ready_sample,
    _keep_persisted_sample,
    _sample_key,
    _sampling_breaker,
    calculate_columns_profiling,
    process_table,
    reset_sampling_breaker,
)
from gsf.utils.sql_identifiers import qualified_name


def _mock_connector(dialect: str = "postgres") -> MagicMock:
    """A mock connector that qualifies names the way a real one would.

    ``calculate_columns_profiling`` delegates qualification to the connector,
    so a bare MagicMock would interpolate a repr into the SQL and every
    assertion on the generated statement would pass vacuously.
    """
    connector = MagicMock()
    connector.dialect = dialect
    connector.qualify.side_effect = lambda schema, table: qualified_name(
        schema, table, dialect=dialect
    )
    return connector


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (np.int64(7), 7),
        (np.float64(1.5), 1.5),
        (np.bool_(True), True),
        (Decimal("10"), 10),
        (Decimal("10.5"), 10.5),
        (datetime(2020, 1, 2, 3, 4, 5), "2020-01-02T03:04:05"),
        (date(2020, 1, 2), "2020-01-02"),
        (b"bytes", "bytes"),
        (["a", np.int64(1)], ["a", 1]),
        ({"k": Decimal("2")}, {"k": 2}),
        ("plain", "plain"),
        (None, None),
    ],
)
def test_json_ready_sample_converts_driver_types(raw: object, expected: object) -> None:
    converted = _json_ready_sample(raw)
    assert converted == expected
    # numpy scalars compare equal to their Python counterparts, so identity of
    # the type is the only thing that shows the unwrap actually happened.
    assert type(converted) is type(expected)


def test_sample_key_separates_values_a_str_key_would_merge() -> None:
    # str() would render all three as "1"/"True" collisions across types.
    assert len({_sample_key(1), _sample_key("1"), _sample_key(True)}) == 3
    # Unhashable values still get a key, and equal ones share it.
    assert _sample_key(["a", "b"]) == _sample_key(["a", "b"])
    assert _sample_key({"k": 1}) != _sample_key({"k": 2})


def test_keep_persisted_sample_caps_text_only() -> None:
    assert _keep_persisted_sample("short") is True
    assert _keep_persisted_sample("x" * 31) is False
    assert _keep_persisted_sample(10**40) is True
    assert _keep_persisted_sample(1.5) is True
    assert _keep_persisted_sample(False) is True
    # Non-scalars are judged by how long they render.
    assert _keep_persisted_sample(["a", "b"]) is True
    assert _keep_persisted_sample(["x" * 40]) is False


@patch("gsf.semantic.visit_enter.store_column_date_formats")
@patch("gsf.semantic.visit_enter.store_column_uniqueness")
@patch("gsf.semantic.visit_enter.store_column_sample_values")
def test_calculate_columns_profiling_quotes_name_with_space(
    mock_store_samples: MagicMock,
    mock_store_unique: MagicMock,
    mock_store_dates: MagicMock,
) -> None:
    # Unquoted "main.Sales Orders" parses as table main.Sales, so the probe
    # raised "no such table" and every column of the table was dropped.
    df = pd.DataFrame({"OrderDate": ["5/31/18", "6/1/18", "12/4/18"]})
    connector = _mock_connector("sqlite")
    connector.execute.return_value = df

    table = {"id": "t1", "name": "Sales Orders", "schema_name": "main"}
    columns = [{"name": "OrderDate", "data_type": "TEXT"}]

    result = calculate_columns_profiling(table, columns, connector)

    sql = connector.execute.call_args_list[0][0][0]
    assert '"main"."Sales Orders"' in sql
    assert result["OrderDate"]["format"] == "M/D/YY"


@patch("gsf.semantic.visit_enter.store_column_date_formats")
@patch("gsf.semantic.visit_enter.store_column_uniqueness")
@patch("gsf.semantic.visit_enter.store_column_sample_values")
def test_profiling_probe_uses_the_connectors_qualification(
    mock_store_samples: MagicMock,
    mock_store_unique: MagicMock,
    mock_store_dates: MagicMock,
) -> None:
    # Three-level engines (Kyuubi/Trino/Databricks) prepend their bound catalog.
    # Building schema.table here instead sent the probe to the default catalog,
    # where the tables do not exist -- 197/197 tables failed, silently.
    connector = MagicMock()
    connector.dialect = "spark"
    connector.qualify.side_effect = lambda schema, table: qualified_name(
        "nvapp", schema, table, dialect="spark"
    )
    connector.execute.return_value = pd.DataFrame({"id": [1, 2, 3]})

    table = {"id": "t1", "name": "tbl_Click", "schema_name": "nvapp_client"}
    columns = [{"name": "id", "data_type": "BIGINT"}]

    calculate_columns_profiling(table, columns, connector)

    connector.qualify.assert_called_once_with("nvapp_client", "tbl_Click")
    sql = connector.execute.call_args_list[0][0][0]
    assert "`nvapp`.`nvapp_client`.`tbl_Click`" in sql


@patch("gsf.semantic.visit_enter.store_column_date_formats")
@patch("gsf.semantic.visit_enter.store_column_uniqueness")
@patch("gsf.semantic.visit_enter.store_column_sample_values")
def test_calculate_columns_profiling_unhashable_values(
    mock_store_samples: MagicMock,
    mock_store_unique: MagicMock,
    mock_store_dates: MagicMock,
) -> None:
    # Postgres array / JSON columns come back as Python lists/dicts, which are
    # unhashable — profiling must not crash on them.
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "tags": [["a", "b"], ["a", "b"]],
            "meta": [{"k": 1}, {"k": 2}],
        }
    )
    connector = _mock_connector()
    connector.execute.return_value = df

    table = {"id": "t1", "name": "orders", "schema_name": "public"}
    columns = [
        {"name": "id", "data_type": "integer"},
        {"name": "tags", "data_type": "ARRAY"},
        {"name": "meta", "data_type": "jsonb"},
    ]

    result = calculate_columns_profiling(table, columns, connector)

    assert set(result) == {"id", "tags", "meta"}
    assert result["tags"]["is_unique"] is False  # ["a","b"] repeated
    assert result["meta"]["is_unique"] is True
    assert result["id"]["is_unique"] is True

    # Containers stay containers rather than being flattened to their repr.
    assert result["tags"]["sample_values"] == [["a", "b"]]
    assert result["meta"]["sample_values"] == [{"k": 1}, {"k": 2}]

    stored = mock_store_samples.call_args[0][1]
    assert stored["id"] == [1, 2]
    assert stored["tags"] == [["a", "b"]]


@patch("gsf.semantic.visit_enter.store_column_date_formats")
@patch("gsf.semantic.visit_enter.store_column_uniqueness")
@patch("gsf.semantic.visit_enter.store_column_sample_values")
def test_calculate_columns_profiling(
    mock_store_samples: MagicMock,
    mock_store_unique: MagicMock,
    mock_store_dates: MagicMock,
) -> None:
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "status": ["open", "open", "closed", "open"],
            "created_at": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"],
            "token": ["a" * 40, "b" * 40, "a" * 40, "c" * 40],
        }
    )
    # status and token each have <5 distinct sample values, so each gets a
    # DISTINCT probe. Return empty so the sample top-N values are kept as-is.
    connector = _mock_connector()
    connector.execute.side_effect = [
        df,
        pd.DataFrame({"status": []}),
        pd.DataFrame({"token": []}),
    ]

    table = {"id": "t1", "name": "orders", "schema_name": "public"}
    columns = [
        {"name": "id", "data_type": "integer"},
        {"name": "status", "data_type": "text"},
        {"name": "created_at", "data_type": "timestamp"},
        {"name": "token", "data_type": "text"},
    ]

    result = calculate_columns_profiling(table, columns, connector)

    sql = connector.execute.call_args_list[0][0][0]
    assert '"public"."orders"' in sql
    assert "LIMIT 1000" in sql
    assert connector.execute.call_count == 3

    # Returned dict includes every column (dates, long strings included),
    # each with its top-5 sample values and is_unique flag.
    assert set(result) == {"id", "status", "created_at", "token"}
    assert result["status"]["sample_values"][0] == "open"
    assert len(result["created_at"]["sample_values"]) == 4
    assert result["id"]["is_unique"] is True
    assert result["status"]["is_unique"] is False
    assert result["created_at"]["is_unique"] is True
    assert result["token"]["is_unique"] is False
    assert result["created_at"]["format"] == "YYYY-MM-DD"
    mock_store_dates.assert_called_once()
    assert mock_store_dates.call_args[0][1] == {"created_at": "YYYY-MM-DD"}

    # Uniqueness persisted for every column.
    uniqueness = mock_store_unique.call_args[0][1]
    assert uniqueness == {
        "id": True,
        "status": False,
        "created_at": True,
        "token": False,
    }

    # Stored sample values exclude the date column and the >30-char token values.
    stored = mock_store_samples.call_args[0][1]
    assert "created_at" not in stored
    assert "token" not in stored
    assert stored["status"][0] == "open"
    assert stored["id"] == [1, 2, 3, 4]
    assert all(type(v) is int for v in stored["id"])


@patch("gsf.semantic.visit_enter.store_column_date_formats")
@patch("gsf.semantic.visit_enter.store_column_uniqueness")
@patch("gsf.semantic.visit_enter.store_column_sample_values")
def test_calculate_columns_profiling_preserves_scalar_types(
    mock_store_samples: MagicMock,
    mock_store_unique: MagicMock,
    mock_store_dates: MagicMock,
) -> None:
    """Numbers and booleans profile as themselves, not as their str() form."""
    df = pd.DataFrame(
        {
            "amount": [10, 20, 10, 30],
            "ratio": [1.5, 2.5, 1.5, 3.5],
            "active": [True, False, True, True],
            # Wider than the 30-char cap once rendered — the cap is about prose,
            # so a number is kept however many digits it has.
            "big": [10**40, 10**41, 10**40, 10**42],
            "label": ["a" * 40, "b", "a" * 40, "c"],
        }
    )
    connector = _mock_connector()
    # Only `label` is text with fewer than 5 distinct values, so it is the one
    # column that triggers a DISTINCT probe; return empty to keep the top-N set.
    connector.execute.side_effect = [df, pd.DataFrame({"label": []})]

    table = {"id": "t1", "name": "orders", "schema_name": "public"}
    columns = [
        {"name": "amount", "data_type": "integer"},
        {"name": "ratio", "data_type": "numeric"},
        {"name": "active", "data_type": "boolean"},
        {"name": "big", "data_type": "numeric"},
        {"name": "label", "data_type": "text"},
    ]

    result = calculate_columns_profiling(table, columns, connector)

    assert result["amount"]["sample_values"] == [10, 20, 30]
    assert result["ratio"]["sample_values"] == [1.5, 2.5, 3.5]
    assert result["active"]["sample_values"] == [True, False]
    assert all(isinstance(v, int) for v in result["amount"]["sample_values"])
    assert all(isinstance(v, float) for v in result["ratio"]["sample_values"])
    assert all(isinstance(v, bool) for v in result["active"]["sample_values"])

    stored = mock_store_samples.call_args[0][1]
    assert stored["amount"] == [10, 20, 30]
    assert stored["ratio"] == [1.5, 2.5, 3.5]
    assert stored["active"] == [True, False]
    # The length cap applies to text only: the huge numbers survive, the
    # 40-char string does not.
    assert stored["big"] == [10**40, 10**41, 10**42]
    assert stored["label"] == ["b", "c"]


@patch("gsf.semantic.visit_enter.store_column_date_formats")
@patch("gsf.semantic.visit_enter.store_column_uniqueness")
@patch("gsf.semantic.visit_enter.store_column_sample_values")
def test_calculate_columns_profiling_distinct_only_when_under_top_n(
    mock_store_samples: MagicMock,
    mock_store_unique: MagicMock,
    mock_store_dates: MagicMock,
) -> None:
    """DISTINCT probes run only for text columns with fewer than 5 sample values."""
    sample_df = pd.DataFrame(
        {
            # 5+ distinct values in the sample → no DISTINCT probe.
            "city": ["a", "b", "c", "d", "e", "a"],
            # Fewer than 5 distinct values → DISTINCT probe for rare enums.
            "status": ["open", "open", "closed", "open", "open", "open"],
        }
    )
    status_distinct_df = pd.DataFrame({"status": ["open", "closed", "banned"]})

    connector = _mock_connector()
    connector.execute.side_effect = [sample_df, status_distinct_df]

    table = {"id": "t1", "name": "orders", "schema_name": "public"}
    columns = [
        {"name": "city", "data_type": "string"},
        {"name": "status", "data_type": "text"},
    ]

    result = calculate_columns_profiling(table, columns, connector)

    assert connector.execute.call_count == 2
    distinct_sql = connector.execute.call_args_list[1][0][0]
    assert "SELECT DISTINCT" in distinct_sql
    assert '"status"' in distinct_sql
    assert '"city"' not in distinct_sql

    assert result["city"]["sample_values"] == ["a", "b", "c", "d", "e"]
    assert "banned" in result["status"]["sample_values"]
    assert "open" in result["status"]["sample_values"]
    assert "closed" in result["status"]["sample_values"]


@patch("gsf.semantic.visit_enter.merge_column_attribute")
@patch("gsf.semantic.visit_enter.merge_term")
@patch("gsf.semantic.visit_enter.suggest_potential_foreign_keys")
@patch("gsf.semantic.visit_enter.extract_term")
def test_process_table_writes_term_and_attributes(
    mock_term: MagicMock,
    mock_fk_suggest: MagicMock,
    mock_merge_term: MagicMock,
    mock_merge_col_attr: MagicMock,
) -> None:
    from gsf.semantic.models import (
        PotentialFkResult,
        TableTermsResult,
        TermAttributeAssignment,
        TermProposal,
    )

    mock_fk_suggest.return_value = PotentialFkResult()
    mock_term.return_value = TableTermsResult(
        terms=[
            TermProposal(
                name="Order",
                description="Order entity",
                attributes=[
                    TermAttributeAssignment(
                        source_column="amount",
                        display_name="Total Amount",
                    )
                ],
            )
        ]
    )

    table = {"id": "t1", "name": "orders", "description": "", "schema_name": "public"}
    ctx = {
        "columns": [{"name": "amount", "data_type": "numeric"}],
        "fks": [],
    }

    process_table(table, ctx, domain_summary=None)

    mock_merge_term.assert_called_once_with("Order", "Order entity", "t1", synonyms=[])
    mock_merge_col_attr.assert_called_once()


@patch("gsf.semantic.visit_enter.store_column_date_formats")
@patch("gsf.semantic.visit_enter.store_column_uniqueness")
@patch("gsf.semantic.visit_enter.store_column_sample_values")
def test_distinct_probe_is_gated_but_the_row_sample_is_not(
    mock_store_samples: MagicMock,
    mock_store_unique: MagicMock,
    mock_store_dates: MagicMock,
) -> None:
    """The setting governs the DISTINCT probes only.

    ``SELECT * ... LIMIT`` is bounded and always runs. The DISTINCT probe is
    the unbounded one -- a full-column scan per column where DISTINCT + LIMIT
    does not early-stop -- so it is the one worth turning off.
    """
    # `status` is non-unique text with fewer than _PROFILING_TOP_N distinct
    # sampled values, which is exactly the shape that triggers a DISTINCT probe.
    df = pd.DataFrame({"status": ["open", "open", "closed"]})
    columns = [{"name": "status", "data_type": "text"}]
    table = {"id": "t1", "name": "orders", "schema_name": "public"}

    off = _mock_connector()
    off.execute.return_value = df
    calculate_columns_profiling(table, columns, off, probe_distinct_values=False)
    off_sql = [c[0][0] for c in off.execute.call_args_list]

    on = _mock_connector()
    on.execute.side_effect = [df, pd.DataFrame({"status": ["banned"]})]
    calculate_columns_profiling(table, columns, on, probe_distinct_values=True)
    on_sql = [c[0][0] for c in on.execute.call_args_list]

    # Both sample the rows; only the enabled one issues the DISTINCT probe.
    assert sum("SELECT * FROM" in s for s in off_sql) == 1
    assert sum("SELECT * FROM" in s for s in on_sql) == 1
    assert not any("SELECT DISTINCT" in s for s in off_sql)
    assert sum("SELECT DISTINCT" in s for s in on_sql) == 1


@patch("gsf.semantic.visit_enter.get_connectors")
@patch("gsf.semantic.visit_enter.merge_column_attribute")
@patch("gsf.semantic.visit_enter.merge_term")
@patch("gsf.semantic.visit_enter.suggest_potential_foreign_keys")
@patch("gsf.semantic.visit_enter.extract_term")
def test_row_sample_runs_even_with_the_setting_off(
    mock_term: MagicMock,
    mock_fk_suggest: MagicMock,
    mock_merge_term: MagicMock,
    mock_merge_col_attr: MagicMock,
    mock_get_connectors: MagicMock,
) -> None:
    """``SELECT * ... LIMIT 1000`` is unconditional, end to end.

    Asserted against the real connector-resolution path (``get_connectors``)
    rather than a patched ``_resolve_connector``, and by recording every SQL
    string the connector is asked to run.
    """
    from gsf.semantic.models import PotentialFkResult, TableTermsResult

    mock_fk_suggest.return_value = PotentialFkResult()
    mock_term.return_value = TableTermsResult(terms=[])

    executed: list[str] = []
    spy = MagicMock()
    spy.database_name = "db"
    spy.dialect = "spark"
    spy.qualify.side_effect = lambda s, t: f"`nvapp`.`{s}`.`{t}`"

    def _record(sql: str, *args: object, **kwargs: object) -> pd.DataFrame:
        executed.append(sql)
        return pd.DataFrame({"amount": [1, 2, 3]})

    spy.execute.side_effect = _record
    mock_get_connectors.return_value = [spy]

    table = {"id": "t1", "name": "orders", "description": "", "schema_name": "public"}
    ctx = {"columns": [{"name": "amount", "data_type": "numeric"}], "fks": []}

    process_table(
        table,
        ctx,
        domain_summary=None,
        database_name="db",
        probe_distinct_values=False,
    )

    assert executed == ["SELECT * FROM `nvapp`.`public`.`orders` LIMIT 1000"]


@patch("gsf.semantic.visit_enter.calculate_columns_profiling")
@patch("gsf.semantic.visit_enter._resolve_connector")
@patch("gsf.semantic.visit_enter.merge_column_attribute")
@patch("gsf.semantic.visit_enter.merge_term")
@patch("gsf.semantic.visit_enter.suggest_potential_foreign_keys")
@patch("gsf.semantic.visit_enter.extract_term")
def test_process_table_profiles_by_default(
    mock_term: MagicMock,
    mock_fk_suggest: MagicMock,
    mock_merge_term: MagicMock,
    mock_merge_col_attr: MagicMock,
    mock_resolve: MagicMock,
    mock_profiling: MagicMock,
) -> None:
    from gsf.semantic.models import PotentialFkResult, TableTermsResult

    mock_fk_suggest.return_value = PotentialFkResult()
    mock_term.return_value = TableTermsResult(terms=[])
    mock_profiling.return_value = {}

    table = {"id": "t1", "name": "orders", "description": "", "schema_name": "public"}
    ctx = {"columns": [{"name": "amount", "data_type": "numeric"}], "fks": []}

    process_table(table, ctx, domain_summary=None, database_name="db")

    mock_profiling.assert_called_once()


@patch("gsf.semantic.visit_enter.merge_column_attribute")
@patch("gsf.semantic.visit_enter.merge_term")
@patch("gsf.semantic.visit_enter.suggest_potential_foreign_keys")
@patch("gsf.semantic.visit_enter.extract_term")
def test_process_table_skips_fk_columns(
    mock_term: MagicMock,
    mock_fk_suggest: MagicMock,
    mock_merge_term: MagicMock,
    mock_merge_col_attr: MagicMock,
) -> None:
    from gsf.semantic.models import (
        PotentialFkResult,
        TableTermsResult,
        TermProposal,
    )

    mock_fk_suggest.return_value = PotentialFkResult()
    mock_term.return_value = TableTermsResult(
        terms=[TermProposal(name="Order", description="", attributes=[])]
    )

    table = {"id": "t1", "name": "orders", "description": "", "schema_name": ""}
    ctx = {
        "columns": [{"name": "customer_id", "data_type": "integer"}],
        "fks": [
            {
                "source_column": "customer_id",
                "target_table": "customers",
                "target_table_id": "t2",
            }
        ],
    }

    process_table(table, ctx, domain_summary=None)

    mock_merge_term.assert_not_called()


@patch("gsf.semantic.visit_enter.merge_column_attribute")
@patch("gsf.semantic.visit_enter.merge_term")
@patch("gsf.semantic.visit_enter.suggest_potential_foreign_keys")
@patch("gsf.semantic.visit_enter.extract_term")
def test_process_table_multiple_terms(
    mock_term: MagicMock,
    mock_fk_suggest: MagicMock,
    mock_merge_term: MagicMock,
    mock_merge_col_attr: MagicMock,
) -> None:
    from gsf.semantic.models import (
        PotentialFkResult,
        TableTermsResult,
        TermAttributeAssignment,
        TermProposal,
    )

    mock_fk_suggest.return_value = PotentialFkResult()
    mock_term.return_value = TableTermsResult(
        terms=[
            TermProposal(
                name="Order",
                description="Core order",
                attributes=[
                    TermAttributeAssignment(
                        source_column="amount", display_name="Total Amount"
                    )
                ],
            ),
            TermProposal(
                name="Audit Metadata",
                description="Audit fields",
                attributes=[
                    TermAttributeAssignment(
                        source_column="created_at", display_name="Created At"
                    )
                ],
            ),
        ]
    )

    table = {"id": "t1", "name": "orders", "description": "", "schema_name": "public"}
    ctx = {
        "columns": [
            {"name": "amount", "data_type": "numeric"},
            {"name": "created_at", "data_type": "timestamp"},
        ],
        "fks": [],
    }

    process_table(table, ctx, domain_summary=None)

    assert mock_merge_term.call_count == 2
    assert mock_merge_col_attr.call_count == 2


# ----------------------------------------------------------------------
# Sample query cap and circuit breaker
# ----------------------------------------------------------------------


class _CappedConnector:
    """A connector whose execute() accepts a timeout, like Kyuubi's."""

    dialect = "spark"
    database_name = "nvdp"
    supports_statement_timeout = True

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, int | None]] = []

    def qualify(self, schema: str, table: str) -> str:
        return f"`{schema}`.`{table}`"

    def execute(self, sql: str, parameters=None, *, timeout_s: int | None = None):
        self.calls.append((sql, timeout_s))
        if self.error is not None:
            raise self.error
        return pd.DataFrame({"id": [1, 2]})


TABLE = {"id": "t1", "name": "elk_log", "schema_name": "raw"}
COLUMNS = [{"name": "id", "data_type": "bigint"}]


@pytest.fixture(autouse=True)
def _reset_breaker():
    reset_sampling_breaker(_CappedConnector.database_name)
    yield
    reset_sampling_breaker(_CappedConnector.database_name)


@patch("gsf.semantic.visit_enter.store_column_date_formats")
@patch("gsf.semantic.visit_enter.store_column_uniqueness")
@patch("gsf.semantic.visit_enter.store_column_sample_values")
def test_sample_query_is_capped_where_the_connector_supports_it(*_mocks) -> None:
    """A bounded sample is seconds on a healthy engine; 900s means it is not."""
    connector = _CappedConnector()

    calculate_columns_profiling(TABLE, COLUMNS, connector)

    assert connector.calls[0][1] == _PROFILING_QUERY_TIMEOUT_SECONDS


@patch("gsf.semantic.visit_enter.store_column_date_formats")
@patch("gsf.semantic.visit_enter.store_column_uniqueness")
@patch("gsf.semantic.visit_enter.store_column_sample_values")
def test_a_connector_without_a_cap_is_called_unchanged(*_mocks) -> None:
    """Passing timeout_s to a connector that has no such parameter would fail."""
    connector = _mock_connector()
    connector.execute.return_value = pd.DataFrame({"id": [1, 2]})

    calculate_columns_profiling(TABLE, COLUMNS, connector)

    assert connector.execute.call_args_list[0].kwargs == {}


def test_repeated_sample_timeouts_disable_sampling_for_the_run() -> None:
    """86 tables x a 120s timeout is three hours that produce nothing."""
    connector = _CappedConnector(TimeoutError("cancelled after 120s"))

    for _ in range(_MAX_CONSECUTIVE_SAMPLE_FAILURES):
        assert calculate_columns_profiling(TABLE, COLUMNS, connector) == {}

    assert _sampling_breaker.is_open(connector)


def test_a_sql_error_on_one_table_does_not_disable_sampling() -> None:
    """A table the engine refuses says nothing about the other 85."""
    connector = _CappedConnector(ValueError("Table or view not found: nope"))

    for _ in range(_MAX_CONSECUTIVE_SAMPLE_FAILURES + 2):
        calculate_columns_profiling(TABLE, COLUMNS, connector)

    assert not _sampling_breaker.is_open(connector)


@patch("gsf.semantic.visit_enter.store_column_date_formats")
@patch("gsf.semantic.visit_enter.store_column_uniqueness")
@patch("gsf.semantic.visit_enter.store_column_sample_values")
def test_one_success_clears_the_failure_streak(*_mocks) -> None:
    """Only *consecutive* failures mean the warehouse itself is gone."""
    failing = _CappedConnector(TimeoutError("cancelled"))
    healthy = _CappedConnector()

    calculate_columns_profiling(TABLE, COLUMNS, failing)
    calculate_columns_profiling(TABLE, COLUMNS, healthy)
    calculate_columns_profiling(TABLE, COLUMNS, failing)
    calculate_columns_profiling(TABLE, COLUMNS, failing)

    # Both connectors share a key (the database name), so the success in the
    # middle is what keeps the breaker closed here.
    assert not _sampling_breaker.is_open(failing)


def test_resetting_one_database_leaves_another_run_alone() -> None:
    """Compilations for two datasources overlap; one must not re-arm the other."""
    dead = _CappedConnector(TimeoutError("cancelled"))

    for _ in range(_MAX_CONSECUTIVE_SAMPLE_FAILURES):
        calculate_columns_profiling(TABLE, COLUMNS, dead)
    assert _sampling_breaker.is_open(dead)

    reset_sampling_breaker("some_other_database")

    assert _sampling_breaker.is_open(dead), (
        "a run starting elsewhere must not hand this run back a dead warehouse"
    )


def test_distinct_probe_timeouts_count_toward_the_breaker() -> None:
    """The probe is a full column scan; silent timeouts burn the cap per column."""
    connector = _CappedConnector()
    for _ in range(_MAX_CONSECUTIVE_SAMPLE_FAILURES):
        assert (
            _distinct_values_if_low_cardinality(
                _CappedConnector(TimeoutError("cancelled")),
                "`raw`.`elk_log`",
                "status",
                25,
            )
            is None
        )

    assert _sampling_breaker.is_open(connector)


@patch("gsf.semantic.visit_enter.calculate_columns_profiling")
@patch("gsf.semantic.visit_enter._resolve_connector")
@patch("gsf.semantic.visit_enter.merge_column_attribute")
@patch("gsf.semantic.visit_enter.merge_term")
@patch("gsf.semantic.visit_enter.suggest_potential_foreign_keys")
@patch("gsf.semantic.visit_enter.extract_term")
def test_a_tripped_breaker_stops_process_table_sampling(
    mock_term: MagicMock,
    mock_fk_suggest: MagicMock,
    mock_merge_term: MagicMock,
    mock_merge_col_attr: MagicMock,
    mock_resolve: MagicMock,
    mock_profiling: MagicMock,
) -> None:
    """Tripping the breaker is only worth anything if it saves the 120s."""
    from gsf.semantic.models import PotentialFkResult, TableTermsResult

    mock_fk_suggest.return_value = PotentialFkResult()
    mock_term.return_value = TableTermsResult(terms=[])
    connector = _CappedConnector(TimeoutError("cancelled"))
    mock_resolve.return_value = connector

    for _ in range(_MAX_CONSECUTIVE_SAMPLE_FAILURES):
        _sampling_breaker.record_failure(connector, TimeoutError("cancelled"))
    assert _sampling_breaker.is_open(connector)

    table = {"id": "t1", "name": "orders", "description": "", "schema_name": "public"}
    ctx = {"columns": [{"name": "amount", "data_type": "numeric"}], "fks": []}
    process_table(table, ctx, domain_summary=None, database_name="nvdp")

    mock_profiling.assert_not_called()


def test_the_breaker_key_matches_however_the_database_is_spelled() -> None:
    """_resolve_connector matches case-insensitively, so the key must too.

    A connector loaded from CONNECTION_STRINGS can spell its database
    differently from the catalog name the run compiles under; a key that
    disagreed with the reset would leave sampling off for every later run.
    """

    class _Shouty(_CappedConnector):
        database_name = "NVDP"

    dead = _Shouty(TimeoutError("cancelled"))
    for _ in range(_MAX_CONSECUTIVE_SAMPLE_FAILURES):
        _sampling_breaker.record_failure(dead, TimeoutError("cancelled"))
    assert _sampling_breaker.is_open(dead)

    reset_sampling_breaker("nvdp")

    assert not _sampling_breaker.is_open(dead)
