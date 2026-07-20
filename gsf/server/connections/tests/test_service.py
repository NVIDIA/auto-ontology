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
