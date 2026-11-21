# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Postgres connection string construction."""

from __future__ import annotations

import pytest

from auto_ontology.infra.postgres import get_postgres_connection_string


def test_password_with_at_sign_is_escaped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_USER", "test")
    monkeypatch.setenv("POSTGRES_PASSWORD", "abc@1234")
    monkeypatch.setenv("POSTGRES_HOST", "127.0.0.1")
    monkeypatch.setenv("POSTGRES_PORT", "5432")
    monkeypatch.setenv("POSTGRES_DATABASE", "test_db")

    conn_str = get_postgres_connection_string()
    assert conn_str == "postgresql://test:abc%401234@127.0.0.1:5432/test_db"


def test_credentials_with_spaces_use_percent_twenty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POSTGRES_USER", "user name")
    monkeypatch.setenv("POSTGRES_PASSWORD", "pass word")
    monkeypatch.setenv("POSTGRES_HOST", "127.0.0.1")
    monkeypatch.setenv("POSTGRES_PORT", "5432")
    monkeypatch.setenv("POSTGRES_DATABASE", "test_db")

    conn_str = get_postgres_connection_string()
    assert conn_str == "postgresql://user%20name:pass%20word@127.0.0.1:5432/test_db"
