# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The catalog write path's single entry point.

Extraction and the write, plus the DataFrame concat below — deliberately a
plain function rather than an operator class, so nothing here has to implement
a library ABC.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from gsf.catalog.extract import extract_tabular_db_data
from gsf.catalog.write import populate_tabular_data

if TYPE_CHECKING:  # pragma: no cover - typing only
    from nemo_retriever.tabular_data.sql_database import SQLDatabase

logger = logging.getLogger(__name__)


def ingest_catalog(connector: "SQLDatabase") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract *connector*'s catalog, write it to the store, return what landed.

    The returned ``(tables_df, columns_df)`` pair is concatenated across every
    ingested :class:`~gsf.catalog.model.schema.Schema` and carries the ids
    assigned to each Table/Column node, so downstream embedding can build its
    text without a round trip back to the store. Per-row ``table_schema`` keeps
    schemas distinguishable.

    Returns a pair of empty DataFrames when there is nothing to ingest, so
    callers can stay straight-line.
    """
    empty = (pd.DataFrame(), pd.DataFrame())
    if connector is None:
        return empty

    data = extract_tabular_db_data(connector)
    if not data:
        return empty

    schemas = populate_tabular_data(data, num_workers=4, dialect=connector.dialect)
    if not schemas:
        return empty

    tables = [s.tables_df for s in schemas.values() if s.tables_df is not None]
    columns = [s.columns_df for s in schemas.values() if s.columns_df is not None]
    tables_df = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    columns_df = pd.concat(columns, ignore_index=True) if columns else pd.DataFrame()
    return tables_df, columns_df
