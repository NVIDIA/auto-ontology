import pandas as pd
import pytest

from gsf.retrieval.kumo import pql_gen
from gsf.retrieval.kumo.pql_gen import (
    _NeighbourhoodMemo,
    _is_context_capacity_error,
    _is_context_size_limit_error,
    _is_transient_exec_error,
    _predict_resilient,
    _resolve_indices,
    _retry_at_full_neighbourhood,
    canonicalize_pql_identifiers,
    extract_pql,
    parse_entity,
)

# The SDK's client-side per-table row cap (kumorfm.rfm.payload.validate_payload_table_rows).
_ROW_LIMIT_ERROR = (
    "Request batch 0 table 'context.related_tables.GPU_ALLOCATIONS' contains "
    "32,000 rows, exceeding the 10,000-row limit"
)


class _Connector:
    def __init__(self) -> None:
        self.sql = ""

    def execute(self, sql: str) -> pd.DataFrame:
        self.sql = sql
        return pd.DataFrame({"JOB_ID": ["job-1"]})


class _UnexpectedConnector:
    def execute(self, _sql: str) -> pd.DataFrame:
        raise AssertionError("source database should not be queried")


def test_extract_pql_ignores_predict_in_explanatory_prose() -> None:
    response = """We need to predict the outcome from the available data.
PREDICT jobs.status FOR EACH jobs.job_id
This query scores every job.
"""

    assert extract_pql(response) == "PREDICT jobs.status FOR EACH jobs.job_id"


def test_extract_pql_rejects_response_without_statement() -> None:
    assert extract_pql("We should predict the likely outcome.") == ""


def test_extract_pql_preserves_fenced_statement() -> None:
    response = """```pql
PREDICT events.status
FOR EACH jobs.job_id
```"""

    assert extract_pql(response) == ("PREDICT events.status\nFOR EACH jobs.job_id")


def test_canonicalize_pql_identifiers_uses_graph_casing() -> None:
    graph_ddl = (
        "JOBS(JOB_ID primary_key, PRIORITY_TIER categorical)  -- PRIMARY KEY (JOB_ID)"
    )

    assert (
        canonicalize_pql_identifiers(
            "PREDICT jobs.priority_tier FOR EACH jobs.job_id",
            graph_ddl,
        )
        == "PREDICT JOBS.PRIORITY_TIER FOR EACH JOBS.JOB_ID"
    )


def test_resolve_indices_queries_canonical_snowflake_identifier() -> None:
    connector = _Connector()
    graph_ddl = (
        "JOBS(JOB_ID primary_key, PRIORITY_TIER categorical)  -- PRIMARY KEY (JOB_ID)"
    )
    pql = canonicalize_pql_identifiers(
        "PREDICT jobs.priority_tier FOR EACH jobs.job_id",
        graph_ddl,
    )

    assert _resolve_indices(
        pql,
        None,
        connector,
        10,
        {"JOBS": '"GPU_FLEET"."JOBS"'},
    ) == ["job-1"]
    assert connector.sql == (
        'SELECT DISTINCT "JOB_ID" FROM "GPU_FLEET"."JOBS" '
        'WHERE "JOB_ID" IS NOT NULL LIMIT 10'
    )


def test_resolve_indices_uses_ids_loaded_into_graph() -> None:
    assert _resolve_indices(
        "PREDICT JOBS.PRIORITY_TIER FOR EACH JOBS.JOB_ID",
        None,
        _UnexpectedConnector(),
        2,
        available_entity_ids={
            "jobs": ["job-in-graph-1", "job-in-graph-2", "job-in-graph-3"]
        },
    ) == ["job-in-graph-1", "job-in-graph-2"]


def test_row_limit_rejection_is_a_context_capacity_error() -> None:
    """It must reach the neighbourhood backoff, not escape into the PQL regenerate loop."""
    assert _is_context_capacity_error(_ROW_LIMIT_ERROR)


def test_row_limit_rejection_steps_down_instead_of_retrying_at_full() -> None:
    """It is deterministic — the same neighbourhood rebuilds the same oversize table every time."""
    assert _is_context_size_limit_error(_ROW_LIMIT_ERROR)
    assert not _retry_at_full_neighbourhood(_ROW_LIMIT_ERROR)


def test_unrelated_errors_are_not_classified_as_row_limit() -> None:
    assert not _is_context_size_limit_error("Failed to parse query")
    assert not _is_context_capacity_error("Failed to parse query")


_NIM_500_ERROR = (
    "An unexpected exception occurred. Please create an issue at "
    "'https://github.com/kumo-ai/kumo-rfm'. Unexpected server error."
)


def test_nim_internal_error_is_classified_as_infra_not_a_bad_query() -> None:
    """A bare HTTP 500 from the NIM must stop the repair loop, not trigger 5 rewrites."""
    assert _is_transient_exec_error(_NIM_500_ERROR)
    # It must NOT look like a capacity problem, or it would walk the neighbourhood ladder.
    assert not _is_context_capacity_error(_NIM_500_ERROR)


def test_neighbourhood_memo_skips_rungs_already_proven_too_big() -> None:
    calls: list[list[int] | None] = []

    def _fail_until_smallest(num_neighbors: list[int] | None) -> str:
        calls.append(num_neighbors)
        if num_neighbors != [8, 8]:
            raise RuntimeError(_ROW_LIMIT_ERROR)
        return "ok"

    memo = _NeighbourhoodMemo()
    assert _predict_resilient(_fail_until_smallest, memo=memo) == "ok"
    first_pass = list(calls)
    assert first_pass == [None, [24, 24], [16, 16], [8, 8]]

    # Second attempt in the same run must not re-pay the three rejected rungs.
    calls.clear()
    assert _predict_resilient(_fail_until_smallest, memo=memo) == "ok"
    assert calls == [[8, 8]]


def test_neighbourhood_memo_is_not_advanced_by_intermittent_gpu_faults() -> None:
    """CUDA/OOM faults are intermittent — the next attempt must still start at full accuracy."""
    state = {"fail": True}

    def _flaky(num_neighbors: list[int] | None) -> str:
        if state["fail"] and num_neighbors is None:
            raise RuntimeError("CUDA error: an illegal memory access was encountered")
        return "ok"

    memo = _NeighbourhoodMemo()
    assert _predict_resilient(_flaky, memo=memo) == "ok"
    assert memo.floor == 0

    state["fail"] = False
    calls: list[list[int] | None] = []

    def _record(num_neighbors: list[int] | None) -> str:
        calls.append(num_neighbors)
        return "ok"

    assert _predict_resilient(_record, memo=memo) == "ok"
    assert calls == [None]


_QUOTED_PQL = (
    "PREDICT COUNT(ORDERS.*, 0, 30, days) = 0 FOR EACH `My People`.`Customer ID`"
)


def test_parse_entity_reads_a_quoted_name_as_the_data_spells_it() -> None:
    """Entity resolution feeds warehouse SQL and the entity-id map, so the
    backticks must not survive parsing."""
    assert parse_entity(_QUOTED_PQL) == ("My People", "Customer ID")


def test_parse_entity_still_reads_a_bare_name() -> None:
    assert parse_entity(
        "PREDICT COUNT(ORDERS.*, 0, 30, days) = 0 FOR EACH PEOPLE.CUSTOMER_ID"
    ) == ("PEOPLE", "CUSTOMER_ID")


def test_canonicalisation_keeps_the_quoting_a_name_needs() -> None:
    ddl = "My People(Customer ID ID, TIER categorical)"
    assert canonicalize_pql_identifiers(_QUOTED_PQL, ddl) == _QUOTED_PQL


def test_static_lint_reads_through_quotes() -> None:
    """The aggregated table is compared against the entity table, so a quoted
    name has to resolve or every quoted query would look cross-table."""
    pql_gen.validate_pql_static(_QUOTED_PQL)

    with pytest.raises(pql_gen.PqlStaticError, match="can only filter columns"):
        pql_gen.validate_pql_static(
            "PREDICT COUNT(ORDERS.* WHERE `My People`.TIER = 'pro', 0, 30, days) "
            "> 0 FOR EACH `My People`.`Customer ID`"
        )


def test_static_lint_accepts_a_quoted_event_table_that_links_to_the_entity() -> None:
    """Graph edges carry bare names, so a backticked aggregation table has to be
    unquoted before the direct-foreign-key check compares the two."""
    pql_gen.validate_pql_static(
        "PREDICT COUNT(`Sales Orders`.*, 0, 30, days) FOR EACH `Store Locations`.StoreID",
        edges=[("Sales Orders", "_StoreID", "Store Locations")],
    )

    with pytest.raises(pql_gen.PqlStaticError, match="no direct foreign key"):
        pql_gen.validate_pql_static(
            "PREDICT COUNT(`Line Items`.*, 0, 30, days) FOR EACH `Store Locations`.StoreID",
            edges=[("Sales Orders", "_StoreID", "Store Locations")],
        )


def test_unquote_and_quote_round_trip() -> None:
    assert pql_gen.unquote_name("`Customer ID`") == "Customer ID"
    assert pql_gen.unquote_name("CUSTOMER_ID") == "CUSTOMER_ID"
    assert pql_gen.quote_name("Customer ID") == "`Customer ID`"
    assert pql_gen.quote_name("CUSTOMER_ID") == "CUSTOMER_ID"
    assert pql_gen.quote_name("*") == "*"
