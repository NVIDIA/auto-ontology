# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Catalog vocabulary — node labels, edge types, and property keys.

The semantic-tier counterparts live in :mod:`auto_ontology.semantic.constants`.
"""

from __future__ import annotations


class Labels:
    """Data-layer node labels."""

    DB = "Database"
    SCHEMA = "Schema"
    TABLE = "Table"
    COLUMN = "Column"
    SQL = "Sql"
    CUSTOM_ANALYSIS = "CustomAnalysis"

    LIST_OF_ALL = [
        DB,
        CUSTOM_ANALYSIS,
        SCHEMA,
        TABLE,
        COLUMN,
        SQL,
    ]


class TableTypes:
    """Canonical ``table_type`` values, in Postgres ``information_schema`` style."""

    VIEW = "view"
    MATERIALIZED_VIEW = "materialized view"
    BASE_TABLE = "base table"

    LIST_OF_ALL = [
        VIEW,
        MATERIALIZED_VIEW,
        BASE_TABLE,
    ]


class Edges:
    """Data-layer relationship types."""

    CONTAINS = "CONTAINS"
    FOREIGN_KEY = "FOREIGN_KEY"
    JOIN = "JOIN"
    UNION = "UNION"
    SQL = "SQL"
    HAS_SQL = "HAS_SQL"


class Props:
    """Edge and node property keys."""

    JOIN = "join"
    UNION = "union"
    SQL_ID = "sql_id"
    ANALYSIS_ID = "analysis_id"
