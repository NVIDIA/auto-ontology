# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Catalog-backed schema and node lookups.

Builds the :class:`Schema` objects consumed by the SQL parser plus generic
``fetch_item_by_id`` helpers.

All direct store access lives in auto_ontology/dal/datasources.py.
This module only keeps the pure-Python ``get_schemas_by_ids`` that assembles
pandas DataFrames into :class:`Schema` objects.
"""

from __future__ import annotations

import logging
import threading
import time

import pandas as pd

from auto_ontology.catalog.model import CatalogNode
from auto_ontology.catalog.constants import Labels
from auto_ontology.catalog.model import Schema
from auto_ontology.catalog.parsers.sqlglot_extractor import SchemaCatalog

from auto_ontology.dal.datasources import (
    fetch_all_schema_ids,
    fetch_catalog_schema_fingerprint,
    fetch_catalog_schema_snapshot,
    fetch_schemas_by_ids,
    fetch_item_by_id,
)

logger = logging.getLogger(__name__)

__all__ = [
    "catalog_schemas_lock",
    "fetch_all_schema_ids",
    "fetch_item_by_id",
    "forget_parsed_nodes",
    "get_catalog_schemas",
    "get_schemas_by_ids",
    "reset_catalog_schemas",
]

#: Held while parsing against what :func:`get_catalog_schemas` returns.
#:
#: That map is shared by every caller in the process, and both it and the
#: ``Schema`` objects in it fill caches on first use (table and column nodes,
#: the mapping ``qualify`` reads). Two threads filling one at once would each
#: build their own copy of a node and disagree about which is *the* node for a
#: table. A parse against the shared map takes milliseconds, so one lock around
#: it costs nothing next to the model call that follows. Reentrant, so a caller
#: already holding it can still ask for the map.
catalog_schemas_lock = threading.RLock()

# The whole-catalog map and the fingerprint of the rows it was built from.
_catalog_snapshot: tuple[str, SchemaCatalog] | None = None


def reset_catalog_schemas() -> None:
    """Forget the cached catalog map, so the next caller builds a fresh one."""
    global _catalog_snapshot
    with catalog_schemas_lock:
        _catalog_snapshot = None


def forget_parsed_nodes(schemas: dict) -> None:
    """Start a parse against the shared map with no nodes left from the last one.

    A ``Schema`` builds a table or column node the first time it is asked for
    one and answers from memory after that. Kept across parses, that memory
    makes a lookup depend on what was parsed before it: a column whose stored
    name carries surrounding quotes fails its first lookup and succeeds on
    every later one, so the columns validation reports for the same SQL would
    differ between the first question and the second.

    Every validation used to get schemas nobody had parsed against. Dropping
    the nodes gives each parse the same starting point and keeps what is
    expensive: the frames, the column index and the id map never change after
    the build. Only the map from :func:`get_catalog_schemas` is touched; a
    map built for one call has no earlier parse to forget.
    """
    if not isinstance(schemas, SchemaCatalog):
        return
    for schema in schemas.values():
        schema.forget_nodes()


def get_catalog_schemas() -> dict:
    """The schema map for the whole catalog, rebuilt only when the catalog changes.

    Equivalent to ``get_schemas_by_ids(fetch_all_schema_ids())``, which is what
    SQL validation used to run before every parse. That reads every column of
    every database and turns the rows into ``Schema`` objects; it is linear in
    the size of the catalog, and a question is validated once per SQL attempt,
    up to eight times. Measured without a database in the way: 30 ms for a
    440-column catalog, 1.5 s at 50,000 columns, 4.6 s at 150,000 -- per
    attempt, on the interpreter lock, so concurrent questions queued behind it.

    Here the rows are read only when
    :func:`~auto_ontology.dal.datasources.fetch_catalog_schema_fingerprint`
    differs from the fingerprint of the rows the cached map was built from.
    The check is a single aggregate, and it is made on every call: a table
    ingested a moment ago is visible to the very next validation, exactly as
    it was before.

    The map is stored under the fingerprint that came back *with its rows*,
    from the same statement, never under the one read beforehand. A commit
    between two separate reads would otherwise file new rows under an old
    fingerprint, and a catalog that later returned to that old state would be
    served the wrong map for as long as it stayed there.

    When the fingerprint cannot be read at all, the map is built as it always
    was and not kept. Anything raised while building propagates and nothing is
    cached, so an empty or unreachable catalog fails the same way it did.

    Hold :data:`catalog_schemas_lock` while parsing against the result, and
    call :func:`forget_parsed_nodes` first.
    """
    global _catalog_snapshot

    try:
        fingerprint = fetch_catalog_schema_fingerprint()
    except Exception:
        logger.warning(
            "Catalog fingerprint unavailable; building the schema map uncached",
            exc_info=True,
        )
        return get_schemas_by_ids(fetch_all_schema_ids())

    with catalog_schemas_lock:
        snapshot = _catalog_snapshot
        if snapshot is not None and snapshot[0] == fingerprint:
            return snapshot[1]
        started = time.perf_counter()
        rows_fingerprint, rows = fetch_catalog_schema_snapshot()
        schemas = SchemaCatalog(_schemas_from_rows(rows))
        _catalog_snapshot = (rows_fingerprint, schemas)
        logger.info(
            "Catalog schema map %s in %.2fs (%d schema(s))",
            "built" if snapshot is None else "rebuilt after a catalog change",
            time.perf_counter() - started,
            len(schemas),
        )
        return schemas


def get_schemas_by_ids(relevant_schemas_ids: list | None = None) -> dict:
    """Assemble :class:`Schema` objects from catalog data.

    Fetches raw column/table rows via :func:`auto_ontology.dal.datasources.fetch_schemas_by_ids`
    then builds the in-memory :class:`Schema` map consumed by the SQL parser.
    """
    before_get_all = time.time()
    data_array = fetch_schemas_by_ids(relevant_schemas_ids)
    logger.info(f"time took to get all data from graph: {time.time() - before_get_all}")
    return _schemas_from_rows(data_array)


def _schemas_from_rows(data_array: list[dict[str, str]]) -> dict:
    """Turn ``fetch_schemas_by_ids``-shaped rows into the ``Schema`` map."""
    before_get_all = time.time()
    data_df = pd.DataFrame(data_array)
    dbs = list(data_df["database_name"].unique())

    schemas = data_df[["database_name", "table_schema"]]
    schemas = schemas.drop_duplicates().to_dict(orient="records")

    all_schemas = {}
    schema_dfs = {}
    dbs_nodes = {}
    for database_name in dbs:
        database_node = CatalogNode(
            name=database_name, label=Labels.DB, props={"name": database_name}
        )
        dbs_nodes[database_name] = database_node

    tables_df = data_df[
        ["database_name", "table_schema", "table_name", "table_id"]
    ].drop_duplicates(subset=["database_name", "table_schema", "table_name"])
    tables_df = tables_df.rename(columns={"table_id": "id"})

    unique_schemas = data_df.table_schema.unique()
    for table_schema in unique_schemas:
        schema_tables_df = tables_df.loc[tables_df["table_schema"] == table_schema]
        schema_dfs[table_schema] = {
            "tables": schema_tables_df.to_dict(orient="records")
        }

    for table_schema in unique_schemas:
        columns_df = data_df.loc[data_df["table_schema"] == table_schema].rename(
            columns={"column_id": "id"}
        )
        schema_dfs[table_schema]["columns"] = columns_df.to_dict(orient="records")

    before_modify_all = time.time()
    for schema in schemas:
        table_schema: str = schema.get("table_schema")
        if not table_schema:
            continue

        schema_database_name: str = schema["database_name"]
        schema_database_node = dbs_nodes[schema_database_name]
        tables_df = pd.DataFrame(schema_dfs[table_schema]["tables"])
        columns_df = pd.DataFrame(schema_dfs[table_schema]["columns"])

        all_schemas[table_schema.lower()] = Schema(
            schema_database_node,
            tables_df,
            columns_df,
            table_schema,
            is_creation_mode=False,
        )
    logger.info(
        f"total time it took to create all schemas nodes: {time.time() - before_modify_all}"
    )
    logger.info(
        f"total time to build schemas from rows: {time.time() - before_get_all}"
    )
    return all_schemas
