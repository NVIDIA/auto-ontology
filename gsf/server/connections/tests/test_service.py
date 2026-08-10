import json
from unittest.mock import patch

from gsf.server.connections import service
from gsf.server.connections.router import _serialize_connection


def _snowflake_connection() -> dict[str, object]:
    return {
        "type": "snowflake",
        "account": "account",
        "warehouse": "warehouse",
        "user": "user",
        "password": "secret",
        "database": "database",
        "schemas": ["GPU_FLEET"],
    }


def test_create_connection_persists_connection_without_vault() -> None:
    connection = _snowflake_connection()

    with (
        patch.object(service, "_database_already_connected", return_value=False),
        patch.object(service, "is_vault_configured", return_value=False),
        patch.object(service, "insert_connection") as insert_connection,
        patch.object(service, "invalidate_connectors_cache"),
        patch.object(service, "refresh_chat_workers"),
        patch.object(service, "trigger_ingest") as trigger_ingest,
    ):
        service.create_connection(connection=connection)

    stored = json.loads(insert_connection.call_args.kwargs["connection"])
    assert stored["password"] == "secret"
    assert stored["schemas"] == ["GPU_FLEET"]
    trigger_ingest.assert_called_once_with(connection)


def test_public_connection_payload_redacts_credentials() -> None:
    serialized = _serialize_connection(
        {
            **_snowflake_connection(),
            "password_env": "SNOWFLAKE_PASSWORD",
        }
    )

    assert serialized["database_name"] == "database"
    assert "password" not in serialized["connection"]
    assert "password_env" not in serialized["connection"]


class _StubConnector:
    def __init__(self, schemas: list[str]) -> None:
        self._schemas = schemas
        self.closed = False

    def get_schemas(self) -> list[str]:
        return self._schemas

    def close(self) -> None:
        self.closed = True


def _databricks_connection(**extra: object) -> dict[str, object]:
    return {
        "type": "databricks",
        "host": "example.databricks.com",
        "http_path": "/sql/1.0/warehouses/w",
        "password": "secret",
        "database": "main",
        **extra,
    }


def test_test_connection_accepts_a_schema_that_exists() -> None:
    connector = _StubConnector(["Sales", "marketing"])

    with (
        patch.object(service, "_database_already_connected", return_value=False),
        patch.object(service, "create_connector", return_value=connector),
    ):
        # Case-insensitive: Databricks reports its own casing.
        schemas = service.test_connection(_databricks_connection(schema="sales"))

    assert schemas == ["Sales", "marketing"]
    assert connector.closed


def test_test_connection_rejects_a_schema_that_does_not_exist() -> None:
    """The UI skips schema selection once a schema is named, so a typo has to fail
    here — otherwise it surfaces later as an ingest that silently finds nothing."""
    connector = _StubConnector(["sales"])

    with (
        patch.object(service, "_database_already_connected", return_value=False),
        patch.object(service, "create_connector", return_value=connector),
    ):
        try:
            service.test_connection(_databricks_connection(schema="typo_schema"))
        except ValueError as exc:
            assert "typo_schema" in str(exc)
        else:
            raise AssertionError("expected a ValueError for the missing schema")

    assert connector.closed


def test_create_connection_drops_the_schema_shorthand() -> None:
    """`schema` is the form's shorthand for `schemas`; storing both would leave two
    sources of truth for what gets ingested."""
    connection = _databricks_connection(schema="sales", schemas=["sales"])

    with (
        patch.object(service, "_database_already_connected", return_value=False),
        patch.object(service, "is_vault_configured", return_value=False),
        patch.object(service, "insert_connection") as insert_connection,
        patch.object(service, "invalidate_connectors_cache"),
        patch.object(service, "refresh_chat_workers"),
        patch.object(service, "trigger_ingest"),
    ):
        service.create_connection(connection=connection)

    stored = json.loads(insert_connection.call_args.kwargs["connection"])
    assert "schema" not in stored
    assert stored["schemas"] == ["sales"]
