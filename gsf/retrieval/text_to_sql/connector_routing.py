# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Connector-routing helpers for the text-to-SQL agents.

Given a list of injected ``SQLDatabase`` connectors and the
``relevant_tables`` carried on ``path_state``, these helpers pick the
connector that owns the tables the SQL was generated against. Used for
prompt-dialect resolution, SQL execution, and (in future) federated-query
detection.
"""

from __future__ import annotations

from typing import Iterable

from nemo_retriever.tabular_data.sql_database import SQLDatabase


def resolve_target_database_name(
    target_db: str,
    connectors: list[SQLDatabase],
) -> str:
    """Return the configured database name matching *target_db*.

    Matching is exact first and then case-insensitive. Unknown identifiers are
    rejected before they can be used as vector-metadata filters.
    """
    requested = target_db.strip()
    database_names = [
        str(database_name)
        for connector in connectors
        if (database_name := getattr(connector, "database_name", None))
    ]

    if requested in database_names:
        return requested

    casefold_matches = [
        database_name
        for database_name in database_names
        if database_name.casefold() == requested.casefold()
    ]
    if len(casefold_matches) == 1:
        return casefold_matches[0]

    available = ", ".join(sorted(database_names)) or "(none)"
    raise ValueError(
        f"target_db {target_db!r} does not match a configured database name. "
        f"Available databases: {available}."
    )


def resolve_connector_from_tables(
    tables: Iterable[dict],
    connectors: list[SQLDatabase],
) -> SQLDatabase | None:
    """Return the connector that owns every relevant table.

    A single connector is unambiguous even when legacy table metadata lacks a
    ``database_name``. With multiple connectors, missing, unknown, or mixed
    database names are rejected instead of silently executing on the first
    configured database.
    """
    if not connectors:
        return None

    db_to_connector: dict[str, SQLDatabase] = {
        str(getattr(c, "database_name", "")): c
        for c in connectors
        if getattr(c, "database_name", None)
    }

    table_database_names: set[str] = set()
    for table in tables or []:
        if not isinstance(table, dict):
            continue
        database_name = str(table.get("database_name") or "").strip()
        if database_name:
            table_database_names.add(database_name)


    if table_database_names:
        database_name = next(iter(table_database_names))
        connector = db_to_connector.get(database_name)
        if connector is None:
            raise ValueError(
                f"No configured connector matches database {database_name!r}."
            )
        return connector

    if len(connectors) == 1:
        return connectors[0]

    raise ValueError(
        "Relevant tables do not identify a database and multiple connectors "
        "are configured; set target_db before executing SQL."
    )


__all__ = ["resolve_connector_from_tables", "resolve_target_database_name"]
