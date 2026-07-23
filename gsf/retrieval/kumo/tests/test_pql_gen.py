import pandas as pd

from gsf.retrieval.kumo.pql_gen import (
    _resolve_indices,
    canonicalize_pql_identifiers,
    extract_pql,
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
