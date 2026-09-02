# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Statistics about stored SQL statements, independent of where they are stored.

Both are pure functions over data already in hand — one reads a node object, the
other filters a DataFrame — so they are shared rather than reimplemented per
backend, for the same reason as :mod:`gsf.catalog.diff`: a second copy is a
second thing to fix.
"""

from __future__ import annotations

import pandas as pd


def get_sql_counters(sql_node):
    """Return ``(total_counter, {month_property: count})`` from a node.

    The per-month counters are collected because the write path still carries
    them, but **nothing can read them**: every reader matches
    ``count_monthly_YYYY_MM`` while the writer produces ``count_{month}_{year}``.
    The store
    ignores the second element for that reason; it is returned so the two
    backends keep the same signature.
    """
    properties = sql_node.get_properties()
    total_counter = properties["total_counter"]
    count_per_month = {
        key: value for key, value in properties.items() if key.startswith("count_")
    }
    return total_counter, count_per_month


def get_candidate_sql_ids(
    tbl_ids: list[str],
    col_ids: list[str],
    nodes_count: int,
    sqls_tbls_df: pd.DataFrame,
    join_count: int = 0,
    union_count: int = 0,
) -> pd.DataFrame:
    """Pre-filter stored SQL by structural shape before comparing it properly.

    Same tables, same leaf columns, same AST size, same number of joins and
    unions. Only what survives all five gates is worth the expensive sqlglot
    structural comparison.
    """
    if sqls_tbls_df.empty:
        return sqls_tbls_df.iloc[0:0]

    tbl_set = set(tbl_ids)
    col_set = set(col_ids)
    mask = (
        (sqls_tbls_df["nodes_count"] == nodes_count)
        & (sqls_tbls_df["join_count"] == join_count)
        & (sqls_tbls_df["union_count"] == union_count)
        & sqls_tbls_df["tbls"].apply(lambda t: set(t) == tbl_set)
        & sqls_tbls_df["cols"].apply(lambda c: set(c) == col_set)
    )
    return sqls_tbls_df.loc[mask]
