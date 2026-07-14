# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable Cypher expression fragments shared across DAL queries."""

from __future__ import annotations

from gsf.semantic.constants import LABEL_COLUMN_ATTRIBUTE, REL_HAS_ATTRIBUTE


def column_description_expr(col_var: str) -> str:
    """Cypher expression yielding a column's description.

    Prefers the description of the connected ColumnAttribute
    (``(col)-[:HAS_ATTRIBUTE]->(:ColumnAttribute)``); falls back to the
    column's own description. Blank descriptions are treated as missing, so an
    empty ColumnAttribute description does not mask a real column description.

    *col_var* is the Cypher variable already bound to the ``Column`` node.
    """
    return (
        "coalesce("
        f"head([({col_var})-[:{REL_HAS_ATTRIBUTE}]->(att:{LABEL_COLUMN_ATTRIBUTE}) "
        'WHERE att.description IS NOT NULL AND trim(att.description) <> "" '
        "| att.description]), "
        f"CASE WHEN {col_var}.description IS NOT NULL "
        f'AND trim({col_var}.description) <> "" '
        f"THEN {col_var}.description ELSE null END)"
    )
