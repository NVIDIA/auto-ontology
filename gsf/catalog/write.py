# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write an extracted catalog into the store, diffing against what is there."""

from datetime import datetime, timezone
import logging
import time

from gsf.catalog.store.db import (
    db_exists,
    update_node_property,
    delete_schema,
    update_diff_from_existing_schema,
)
from gsf.catalog.store.indexes import add_indices
from concurrent.futures import ThreadPoolExecutor
from gsf.catalog.parsers import schemas_parser
from gsf.catalog.sql_parse import populate_queries
from gsf.catalog.store.schemas import (
    get_schemas_ids_and_names,
    add_fks,
    add_pks,
    delete_old_fks,
    reset_pks,
)
from gsf.catalog.services.schema import add_schema
from gsf.dal.session import write_transaction

logger = logging.getLogger(__name__)


def populate_tabular_data(data, num_workers, dialect):
    logger.info("Using Dialect: " + dialect)

    add_indices()

    tables_df = data["tables"]
    columns_df = data["columns"]

    if tables_df is None or tables_df.empty:
        logger.warning("No tables found in source database; skipping graph population.")
        return {}

    database = data["database_name"]
    logger.info(f"Started parsing db {database}.")

    all_schemas = populate_db(tables_df, columns_df, database, num_workers)

    if "fks" in data:
        populate_fks(fks=data["fks"], database_name=database)

    if "pks" in data:
        populate_pks(pks=data["pks"], database_name=database)

    if "queries" in data:
        populate_queries(all_schemas, data["queries"], num_workers, [dialect])

    return all_schemas


def populate_db(tables_df, columns_df, database, num_workers):
    schemas, db_node = schemas_parser.parse_df(
        tables_df, columns_df, database_name=database
    )
    existing_db_id, loaded = db_exists(db_node)

    latest_timestamp = datetime.now(timezone.utc).replace(microsecond=0)

    if existing_db_id is None or not loaded:
        if existing_db_id is not None:
            db_node.replace_id(existing_db_id)

        before_adding_schemas = time.time()
        for schema_name, schema in schemas.items():
            add_schema(schema, latest_timestamp, num_workers)
            logger.info(f"Added schema {schema_name} to db.")

        update_node_property("db", str(db_node.get_id()), {"pulled": latest_timestamp})

        logger.info(f"Time took to add schemas:{time.time() - before_adding_schemas}")
        return schemas

    before_adding_schema = time.time()
    existing_schemas = get_schemas_ids_and_names(existing_db_id)
    existing_schema_names = [s["schema_name"].lower() for s in existing_schemas]
    new_schemas = schemas.keys()
    schemas_to_add = [
        schema[1]
        for schema in schemas.items()
        if schema[0] in (set(new_schemas) - set(existing_schema_names))
    ]
    for schema in schemas_to_add:
        schema.get_db_node().replace_id(existing_db_id)
        add_schema(schema, latest_timestamp, num_workers)
        logger.info(f"Added schema {schema.get_schema_name()} to db.")

    schemas_to_update = [
        schema[1]
        for schema in schemas.items()
        if schema[0]
        in (set(new_schemas) - (set(new_schemas) - set(existing_schema_names)))
    ]
    for schema in schemas_to_update:
        schema.get_db_node().replace_id(existing_db_id)
    with ThreadPoolExecutor(num_workers) as executor:
        # list() is load-bearing: executor.map returns a lazy iterator, and an
        # exception raised in a worker is stored in its future and re-raised
        # only when that future is consumed. Left unconsumed, every failure to
        # update a schema was discarded in silence -- which is how the column
        # diff could raise on every single re-ingest without anyone noticing.
        list(
            executor.map(
                lambda schema: _update_schema(schema, latest_timestamp),
                schemas_to_update,
            )
        )

    # delete existing - new
    schemas_to_delete = [
        schema_name.lower()
        for schema_name in existing_schema_names
        if schema_name.lower() in (set(existing_schema_names) - set(new_schemas))
    ]
    schemas_ids_to_delete = [
        s["schema_id"]
        for s in existing_schemas
        if s["schema_name"].lower() in schemas_to_delete
    ]
    schemas_props_to_delete = [
        {"id": s["schema_id"], "name": s["schema_name"]}
        for s in existing_schemas
        if s["schema_name"].lower() in schemas_to_delete
    ]
    logger.info(f"Deleting schemas: {[s['name'] for s in schemas_props_to_delete]}")
    for schema_id in schemas_ids_to_delete:
        delete_schema(schema_id)

    logger.info(f"Time took to update schemas:{time.time() - before_adding_schema}")

    update_node_property("db", existing_db_id, {"pulled": latest_timestamp})
    return schemas


def populate_fks(fks, database_name: str):
    logger.info("Adding FKs.")
    last_seen = datetime.now(timezone.utc)
    # One unit: between the insert and the sweep, the catalog holds both the
    # new FKs and the superseded ones. Failing there is less destructive than
    # the pk case below -- a re-ingest corrects it -- but the window is the
    # same and so is the fix.
    with write_transaction():
        add_fks(fks, last_seen, database_name)
        delete_old_fks(last_seen, database_name)


def populate_pks(pks, database_name: str):
    logger.info("Adding PKs.")
    # `reset_pks` NULLs the pk of every table in the database before `add_pks`
    # writes the new ones. Autocommitted separately, a failure in between --
    # a dropped connection, a bad row -- leaves *every* table with pk = NULL
    # and nothing to recover it but a full re-ingest. The prediction graph keys
    # its entities on pk, so those tables silently stop being usable.
    with write_transaction():
        reset_pks(database_name)
        add_pks(pks, database_name)


def _update_schema(schema, latest_timestamp):
    update_diff_from_existing_schema(schema, latest_timestamp)
    logger.info(f"Updated schema {schema.get_schema_name()} to db.")
