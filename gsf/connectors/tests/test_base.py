# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``SQLDatabase`` connector contract.

The abstract surface is the actual interface every connector implements, so
it is pinned here: dropping a method from the ABC would silently let a
connector ship without it and fail at ingestion time instead of at import.
"""

from __future__ import annotations

import inspect
import pathlib

import pandas as pd
import pytest

from gsf import connectors
from gsf.connectors.base import SQLDatabase

GSF_ROOT = pathlib.Path(__file__).resolve().parents[2]

_REQUIRED = {
    "__init__",
    "execute",
    "get_tables",
    "get_columns",
    "get_queries",
    "get_views",
    "get_pks",
    "get_fks",
    "dialect",
    "database_name",
    "close",
}


def test_abstract_surface_is_exactly_the_documented_contract() -> None:
    assert SQLDatabase.__abstractmethods__ == frozenset(_REQUIRED)


def test_cannot_instantiate_without_implementing_everything() -> None:
    class Partial(SQLDatabase):
        def __init__(self, connection_string: str) -> None: ...

    with pytest.raises(TypeError):
        Partial("x")  # type: ignore[abstract]


class _Fake(SQLDatabase):
    """Minimal concrete implementation used to exercise the base class."""

    def __init__(self, connection_string: str = "x") -> None:
        self.closed = False

    def execute(self, sql: str, parameters: list | None = None) -> pd.DataFrame:
        return pd.DataFrame()

    def get_tables(self) -> pd.DataFrame:
        return pd.DataFrame()

    def get_columns(self) -> pd.DataFrame:
        return pd.DataFrame()

    def get_queries(self, hours: int = 24) -> pd.DataFrame:
        return pd.DataFrame()

    def get_views(self) -> pd.DataFrame:
        return pd.DataFrame()

    def get_pks(self) -> pd.DataFrame:
        return pd.DataFrame()

    def get_fks(self) -> pd.DataFrame:
        return pd.DataFrame()

    @property
    def dialect(self) -> str:
        return "fake"

    @property
    def database_name(self) -> str:
        return "fake_db"

    def close(self) -> None:
        self.closed = True


def test_context_manager_closes_on_exit() -> None:
    """``__enter__``/``__exit__`` are the only behaviour the ABC supplies."""
    with _Fake() as db:
        assert db.closed is False
    assert db.closed is True


def test_context_manager_closes_even_when_the_body_raises() -> None:
    db = _Fake()
    with pytest.raises(ValueError):
        with db:
            raise ValueError("boom")
    assert db.closed is True


def test_get_queries_defaults_to_24_hours() -> None:
    """Callers rely on the default; connectors must not redefine it."""
    assert inspect.signature(SQLDatabase.get_queries).parameters["hours"].default == 24


def test_every_shipped_connector_subclasses_the_base() -> None:
    shipped = [
        connectors.DatabricksDatabase,
        connectors.DuckDBDatabase,
        connectors.HeavyDBDatabase,
        connectors.PostgresDatabase,
        connectors.SnowflakeDatabase,
    ]
    for cls in shipped:
        assert issubclass(cls, SQLDatabase), cls.__name__
        assert not cls.__abstractmethods__, f"{cls.__name__} is still abstract"


def test_the_base_is_re_exported_from_the_package() -> None:
    assert connectors.SQLDatabase is SQLDatabase
    assert "SQLDatabase" in connectors.__all__
