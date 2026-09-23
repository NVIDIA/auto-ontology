import json
from unittest.mock import patch

from auto_ontology.server.connections import service
from auto_ontology.server.connections.router import _serialize_connection


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


def _kyuubi_connection() -> dict[str, object]:
    return {
        "type": "kyuubi",
        "host": "hive.example.com",
        "port": "10000",
        "user": "user",
        "database": "database",
        "truststore": "/etc/ssl/ca.jks",
        "truststore_password": "secret",
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


def test_public_connection_payload_redacts_the_private_key() -> None:
    serialized = _serialize_connection(
        {
            **_snowflake_connection(),
            "private_key": "-----BEGIN PRIVATE KEY-----\nsecret\n",
            "private_key_passphrase": "phrase",
        }
    )

    assert "private_key" not in serialized["connection"]
    assert "private_key_passphrase" not in serialized["connection"]
    assert "BEGIN PRIVATE KEY" not in json.dumps(serialized)


def test_public_connection_payload_redacts_the_truststore_password() -> None:
    """Kyuubi's truststore password unlocks the keystore, so it is a credential
    like any other -- only the truststore path itself is safe to show."""
    serialized = _serialize_connection(_kyuubi_connection())

    assert "truststore_password" not in serialized["connection"]
    assert serialized["connection"]["truststore"] == "/etc/ssl/ca.jks"


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


def test_test_connection_allows_the_connection_an_edit_replaces() -> None:
    """Without `replacing`, re-testing an existing connection fails the duplicate
    check — an edit could never be validated before saving."""
    connector = _StubConnector(["sales"])
    stored = _databricks_connection()

    with (
        patch.object(service, "_stored_connection", return_value=stored),
        patch.object(service, "create_connector", return_value=connector),
    ):
        schemas = service.test_connection(
            _databricks_connection(password=""), replacing="main"
        )

    assert schemas == ["sales"]


def test_test_connection_rejects_renaming_the_database_it_replaces() -> None:
    with patch.object(
        service, "_stored_connection", return_value=_databricks_connection()
    ):
        try:
            service.test_connection(
                _databricks_connection(database="other"), replacing="main"
            )
        except ValueError as exc:
            assert "cannot be changed" in str(exc)
        else:
            raise AssertionError("expected a ValueError for the renamed database")


def test_update_connection_keeps_the_stored_secret_when_left_blank() -> None:
    """The UI never receives credentials, so it cannot send them back; a blank
    field means "unchanged" rather than "clear it"."""
    stored = _snowflake_connection()

    with (
        patch.object(service, "_stored_connection", return_value=stored),
        patch.object(service, "is_vault_configured", return_value=False),
        patch.object(service, "insert_connection") as insert_connection,
        patch.object(service, "_refresh_connection_caches"),
        patch.object(service, "trigger_ingest") as trigger_ingest,
    ):
        service.update_connection(
            database_name="database",
            connection={**_snowflake_connection(), "password": "", "user": "renamed"},
        )

    updated = json.loads(insert_connection.call_args.kwargs["connection"])
    assert updated["password"] == "secret"
    assert updated["user"] == "renamed"
    # The schema allowlist is untouched, so the ingested graph is still correct.
    trigger_ingest.assert_not_called()


def test_update_connection_keeps_the_stored_truststore_password() -> None:
    """Redacted on the way out like every other credential, so the edit form has
    none to send back and a blank field must not wipe it."""
    with (
        patch.object(service, "_stored_connection", return_value=_kyuubi_connection()),
        patch.object(service, "is_vault_configured", return_value=False),
        patch.object(service, "insert_connection") as insert_connection,
        patch.object(service, "_refresh_connection_caches"),
        patch.object(service, "trigger_ingest"),
    ):
        service.update_connection(
            database_name="database",
            connection={
                **_kyuubi_connection(),
                "truststore_password": "",
                "host": "hive2.example.com",
            },
        )

    updated = json.loads(insert_connection.call_args.kwargs["connection"])
    assert updated["truststore_password"] == "secret"
    assert updated["host"] == "hive2.example.com"


def test_update_connection_reingests_even_when_a_cache_refresh_fails() -> None:
    """Caches are refreshed after the write lands, so a failure there must not
    swallow the ingest: the new allowlist is already stored, and a retry would
    compare it against itself and skip the ingest for good."""
    with (
        patch.object(
            service, "_stored_connection", return_value=_snowflake_connection()
        ),
        patch.object(service, "is_vault_configured", return_value=False),
        patch.object(service, "insert_connection"),
        patch.object(service, "invalidate_connectors_cache"),
        patch.object(
            service, "refresh_chat_workers", side_effect=RuntimeError("no workers")
        ),
        patch.object(service, "trigger_ingest") as trigger_ingest,
    ):
        service.update_connection(
            database_name="database",
            connection={**_snowflake_connection(), "schemas": ["GPU_FLEET", "BILLING"]},
        )

    trigger_ingest.assert_called_once()


def test_update_connection_reingests_when_the_schema_allowlist_changes() -> None:
    with (
        patch.object(
            service, "_stored_connection", return_value=_snowflake_connection()
        ),
        patch.object(service, "is_vault_configured", return_value=False),
        patch.object(service, "insert_connection"),
        patch.object(service, "_refresh_connection_caches"),
        patch.object(service, "trigger_ingest") as trigger_ingest,
    ):
        service.update_connection(
            database_name="database",
            connection={**_snowflake_connection(), "schemas": ["GPU_FLEET", "BILLING"]},
        )

    trigger_ingest.assert_called_once()


def test_update_connection_rejects_changing_the_connector_type() -> None:
    """The ingested graph was built by the stored connector, so pointing the same
    catalog row at a different driver is a new connection, not an edit."""
    with patch.object(
        service, "_stored_connection", return_value=_snowflake_connection()
    ):
        try:
            service.update_connection(
                database_name="database",
                connection={**_snowflake_connection(), "type": "postgresql"},
            )
        except ValueError as exc:
            assert "cannot be changed" in str(exc)
        else:
            raise AssertionError("expected a ValueError for the changed type")


def test_update_connection_reports_an_unknown_database_as_missing() -> None:
    with patch.object(service, "_stored_connection", return_value=None):
        try:
            service.update_connection(
                database_name="ghost", connection=_snowflake_connection()
            )
        except LookupError as exc:
            assert "ghost" in str(exc)
        else:
            raise AssertionError("expected a LookupError for the unknown database")


def test_delete_connection_clears_the_stored_record() -> None:
    """The record has to go from this process. Leaving it to the ingestion
    service meant a delete did nothing whenever that service was unreachable,
    and the connection came back on the next page load."""
    with (
        patch.object(
            service, "_stored_connection", return_value=_snowflake_connection()
        ),
        patch.object(service, "is_vault_configured", return_value=False),
        patch.object(service, "clear_connection") as clear_connection,
        patch.object(service, "_refresh_connection_caches"),
        patch.object(service, "trigger_reset") as trigger_reset,
    ):
        result = service.delete_connection("database")

    clear_connection.assert_called_once_with(database_name="database")
    trigger_reset.assert_called_once_with("database")
    assert result == {"database_name": "database"}


def test_delete_connection_removes_the_vault_secret_before_the_row() -> None:
    """`list_connections` prefers a Vault secret over the row, so clearing the
    row first would resurrect the connection if the secret delete then failed."""
    calls: list[str] = []

    with (
        patch.object(
            service, "_stored_connection", return_value=_snowflake_connection()
        ),
        patch.object(service, "is_vault_configured", return_value=True),
        patch.object(
            service, "delete_secrets", side_effect=lambda *_: calls.append("vault")
        ),
        patch.object(
            service, "clear_connection", side_effect=lambda **_: calls.append("row")
        ),
        patch.object(service, "_refresh_connection_caches"),
        patch.object(service, "trigger_reset"),
    ):
        service.delete_connection("database")

    assert calls == ["vault", "row"]


def test_delete_connection_reports_an_unknown_database_as_missing() -> None:
    with (
        patch.object(service, "_stored_connection", return_value=None),
        patch.object(service, "clear_connection") as clear_connection,
        patch.object(service, "trigger_reset") as trigger_reset,
    ):
        assert service.delete_connection("ghost") is None

    clear_connection.assert_not_called()
    trigger_reset.assert_not_called()


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
