# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Role lookup fails closed, except when ownership cleanup must see the outage."""

from __future__ import annotations

import pytest

from auto_ontology.server.users import postgres_dal


def _down(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError("database unavailable")


def _unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        postgres_dal, "get_postgres_connection_string", lambda: "postgresql://unused"
    )
    monkeypatch.setattr(postgres_dal.psycopg, "connect", _down)


def test_a_role_lookup_failure_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    _unavailable(monkeypatch)

    assert postgres_dal.get_user_role("alice") is None
    assert postgres_dal.is_admin("alice") is False


def test_a_strict_role_lookup_reports_the_database_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _unavailable(monkeypatch)

    with pytest.raises(RuntimeError, match="database unavailable"):
        postgres_dal.get_user_role("alice", strict=True)
