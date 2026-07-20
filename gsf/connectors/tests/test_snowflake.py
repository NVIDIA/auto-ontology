import pandas as pd
from pytest import MonkeyPatch

from gsf.connectors.snowflake import SnowflakeDatabase


def _connection_string(query: str = "") -> str:
    suffix = f"&{query}" if query else ""
    return f"snowflake://user:password@account?warehouse=warehouse&database=db{suffix}"


def _tables() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "table_schema": ["GPU_FLEET", "GPU_FLEET", "CRM"],
            "table_name": ["GPUS", "JOBS", "CUSTOMERS"],
        }
    )


def test_url_schema_filters_introspection() -> None:
    database = SnowflakeDatabase(_connection_string("schema=gpu_fleet"))

    filtered = database._filter_by_schema(_tables())

    assert filtered["table_name"].tolist() == ["GPUS", "JOBS"]


def test_explicit_schema_selection_overrides_url_schema() -> None:
    database = SnowflakeDatabase(
        _connection_string("schema=gpu_fleet"),
        schemas=["crm"],
    )

    filtered = database._filter_by_schema(_tables())

    assert filtered["table_name"].tolist() == ["CUSTOMERS"]


def test_missing_schema_keeps_all_visible_schemas() -> None:
    database = SnowflakeDatabase(_connection_string())

    filtered = database._filter_by_schema(_tables())

    assert filtered.equals(_tables())


def test_query_history_excludes_blank_query_text(monkeypatch: MonkeyPatch) -> None:
    database = SnowflakeDatabase(_connection_string())
    captured_sql = ""

    def execute(sql: str) -> pd.DataFrame:
        nonlocal captured_sql
        captured_sql = sql
        return pd.DataFrame(columns=["end_time", "query_text"])

    monkeypatch.setattr(database, "execute", execute)

    database.get_queries()

    assert "NULLIF(TRIM(QUERY_TEXT), '') IS NOT NULL" in captured_sql
