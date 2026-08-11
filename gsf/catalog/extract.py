# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pull schema entities out of a SQLDatabase connector.

Forked in Phase 1 of the drop-Neo4j refactor
(``docs/refactor/drop-neo4j/PLAN.md``) from::

    nemo_retriever.tabular_data.ingestion.extract_data

(NeMo-Retriever, Apache-2.0). Two deliberate departures from a verbatim copy,
both recorded in ``DECISIONS.md``:

* ``extract_tabular_db_data`` takes the **connector** rather than a
  ``TabularExtractParams``. The params object is a library type carrying
  nothing this path reads except ``.connector``, and keeping it would leave a
  library dependency in code that is otherwise storage- and library-agnostic.
* ``store_relational_db_in_neo4j`` is gone. It was a two-line forwarder to
  :func:`gsf.catalog.write.populate_tabular_data`, which
  :func:`gsf.catalog.ingest.ingest_catalog` now calls directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from gsf.catalog.normalize import normalize_columns, normalize_tables

if TYPE_CHECKING:  # pragma: no cover - typing only
    from nemo_retriever.tabular_data.sql_database import SQLDatabase


def create_dataframe(connector: "SQLDatabase"):
    """Extract raw schema DataFrames from any SQLDatabase connector."""
    tables = connector.get_tables()
    columns = connector.get_columns()
    views = connector.get_views()
    queries = connector.get_queries()
    pks = connector.get_pks()
    fks = connector.get_fks()
    return tables, columns, views, queries, pks, fks


def data_for_populate_tabular(connector: "SQLDatabase") -> dict[str, Any]:
    """Build the ``data`` dict ``populate_tabular_data()`` expects."""
    tables, columns, views, queries, pks, fks = create_dataframe(connector)
    tables = normalize_tables(tables)
    columns = normalize_columns(columns)
    data = {
        "database_name": connector.database_name,
        "tables": tables,
        "columns": columns,
        "views": views,
        "pks": pks,
        "fks": fks,
        "queries": queries,
    }
    # queries is not used by populate_tabular_data(); include if needed elsewhere
    return data


def extract_tabular_db_data(connector: "SQLDatabase | None" = None) -> dict[str, Any]:
    """Step 1 — pull schema entities from the relational DB into a data dict.

    Returns a data dict with keys ``database_name``, ``tables``, ``columns``,
    ``views``, ``pks``, ``fks`` and ``queries``, or ``{}`` when *connector* is
    ``None``.
    """
    if connector is None:
        return {}
    return data_for_populate_tabular(connector)
