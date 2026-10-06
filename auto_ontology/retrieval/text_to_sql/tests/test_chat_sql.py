# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chat SQL logging reports the statement and the credential behind it."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from pytest import LogCaptureFixture

from auto_ontology.connectors.databricks import AUTH_SSO_FEDERATION, AUTH_STORED_TOKEN
from auto_ontology.retrieval.text_to_sql import chat_sql
from auto_ontology.retrieval.text_to_sql.chat_sql import (
    CHAT_STATEMENT_TIMEOUT_S,
    _MAX_LOGGED_SQL_CHARS,
    execute_chat_sql,
    log_chat_sql,
    truncate_sql,
)

_LOGGER = "auto_ontology.retrieval.text_to_sql.chat_sql"


@pytest.fixture
def configured_timeout(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Stub the settings read; append to the list to change the stored value."""
    stored = [CHAT_STATEMENT_TIMEOUT_S]
    monkeypatch.setattr(chat_sql, "get_sql_query_timeout_seconds", lambda: stored[-1])
    return stored


def test_logs_sso_federated_credential(caplog: LogCaptureFixture) -> None:
    connector = SimpleNamespace(database_name="kdc_ca1", auth_mode=AUTH_SSO_FEDERATION)

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        log_chat_sql(connector, "SELECT 1")

    assert f"[kdc_ca1] Executing chat SQL as {AUTH_SSO_FEDERATION}:\nSELECT 1" in [
        r.getMessage() for r in caplog.records
    ]


def test_logs_stored_token_credential(caplog: LogCaptureFixture) -> None:
    connector = SimpleNamespace(database_name="kdc_ca1", auth_mode=AUTH_STORED_TOKEN)

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        log_chat_sql(connector, "SELECT 1")

    assert f"[kdc_ca1] Executing chat SQL as {AUTH_STORED_TOKEN}:\nSELECT 1" in [
        r.getMessage() for r in caplog.records
    ]


def test_omits_credential_for_connectors_without_auth_mode(
    caplog: LogCaptureFixture,
) -> None:
    """Postgres et al. have one fixed credential — don't imply otherwise."""
    connector = SimpleNamespace(database_name="mydb")

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        log_chat_sql(connector, "SELECT 1")

    messages = [r.getMessage() for r in caplog.records]
    assert "[mydb] Executing chat SQL:\nSELECT 1" in messages
    assert not any("as the" in m for m in messages)


def test_probe_queries_are_labelled(caplog: LogCaptureFixture) -> None:
    connector = SimpleNamespace(database_name="kdc_ca1", auth_mode=AUTH_STORED_TOKEN)

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        log_chat_sql(connector, "SELECT 1", kind="probe SQL")

    assert any("Executing probe SQL as" in r.getMessage() for r in caplog.records)


def test_unknown_database_name_does_not_crash(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        log_chat_sql(SimpleNamespace(), "SELECT 1")

    assert "[unknown] Executing chat SQL:\nSELECT 1" in [
        r.getMessage() for r in caplog.records
    ]


def test_long_sql_is_truncated() -> None:
    assert truncate_sql("SELECT 1") == "SELECT 1"

    truncated = truncate_sql("x" * (_MAX_LOGGED_SQL_CHARS + 50))
    assert truncated.startswith("x" * 100)
    assert "[50 more chars]" in truncated


class _Recorder:
    """Connector stub recording how ``execute`` was called."""

    database_name = "kdc_ca1"

    def __init__(self, supports_timeout: bool) -> None:
        self.supports_statement_timeout = supports_timeout
        self.calls: list[tuple[str, dict]] = []

    def execute(self, sql: str, **kwargs: object) -> str:
        self.calls.append((sql, kwargs))
        return "df"


def test_chat_sql_is_capped_at_30_seconds(configured_timeout: list[int]) -> None:
    connector = _Recorder(supports_timeout=True)

    assert execute_chat_sql(connector, "SELECT 1") == "df"
    assert connector.calls == [("SELECT 1", {"timeout_s": CHAT_STATEMENT_TIMEOUT_S})]
    assert CHAT_STATEMENT_TIMEOUT_S == 30


def test_chat_sql_uses_the_configured_timeout(configured_timeout: list[int]) -> None:
    """Read per statement, so a change in Agent Settings applies to the next one."""
    connector = _Recorder(supports_timeout=True)

    configured_timeout.append(90)
    execute_chat_sql(connector, "SELECT 1")
    configured_timeout.append(5)
    execute_chat_sql(connector, "SELECT 2")

    assert [kwargs for _, kwargs in connector.calls] == [
        {"timeout_s": 90},
        {"timeout_s": 5},
    ]


def test_without_settings_the_default_is_30_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No frontend means no ``configurations`` table; the read must fall back."""

    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("relation does not exist")

    monkeypatch.setattr(
        "auto_ontology.infra.feature_flags._read_configuration_row", _boom
    )
    connector = _Recorder(supports_timeout=True)

    execute_chat_sql(connector, "SELECT 1")

    assert connector.calls == [("SELECT 1", {"timeout_s": 30})]


def test_connectors_without_timeout_support_are_called_unchanged() -> None:
    """A connector without a cap must not get an argument it would reject."""
    connector = _Recorder(supports_timeout=False)

    execute_chat_sql(connector, "SELECT 1")

    assert connector.calls == [("SELECT 1", {})]


def test_probe_sql_is_also_capped(configured_timeout: list[int]) -> None:
    connector = _Recorder(supports_timeout=True)

    execute_chat_sql(connector, "SELECT 1", kind="probe SQL")

    assert connector.calls[0][1] == {"timeout_s": CHAT_STATEMENT_TIMEOUT_S}


def test_timeout_can_be_disabled_explicitly() -> None:
    connector = _Recorder(supports_timeout=True)

    execute_chat_sql(connector, "SELECT 1", timeout_s=None)

    assert connector.calls == [("SELECT 1", {})]
