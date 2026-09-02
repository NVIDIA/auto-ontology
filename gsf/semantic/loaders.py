"""Read metadata layer (Table, Column, fk, join) from the store.

All direct store access lives in gsf/dal/datasources.py.
This module only keeps the pure-Python helper build_tables_index.
"""

from __future__ import annotations

from typing import Any

from gsf.dal.datasources import fetch_sorted_tables

__all__ = ["build_tables_index", "fetch_sorted_tables"]


def build_tables_index() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """All tables indexed by name (first wins if names collide across schemas)."""
    tables = fetch_sorted_tables()
    by_name: dict[str, dict[str, Any]] = {}
    for table in tables:
        by_name.setdefault(table["name"], table)
    return tables, by_name
