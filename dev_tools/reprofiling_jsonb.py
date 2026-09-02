# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Targeted re-profiling for tables that contain JSONB columns.

After the fix in gsf/semantic/visit_enter.py, JSONB columns now store their
key names (instead of nothing) in Column.sample_values.  This script finds
every already-ingested table that has at least one jsonb column in Neo4j and
re-runs only the column profiling step for those tables, updating Neo4j.

The SQL-generation prompt picks up sample_values from Neo4j at query time
(via the back-fill mechanism in candidates_preparation.py), so the fix is
effective immediately after this script completes — no VDB re-embedding needed
for the SQL prompt.  VDB column embeddings will still be stale; run a full
semantic compile later if you want those refreshed too.

Usage (from repo root)::

    uv run python -m dev_tools.reprofiling_jsonb \\
        --connection postgresql://user:pass@localhost:5432/mydb \\
        --connection postgresql://user:pass@localhost:5432/otherdb
        ...
"""

from __future__ import annotations

import argparse
import logging

from gsf.connectors.registry import create_connector
from gsf.dal.datasources import (
    fetch_table_context,
    get_neo4j_conn,
)
from gsf.semantic.visit_enter import calculate_columns_profiling

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("reprofiling_jsonb")


def _find_jsonb_tables_in_neo4j(database_name: str) -> list[dict]:
    """Return Neo4j table rows that contain at least one jsonb column."""
    from nemo_retriever.tabular_data.ingestion.model.reserved_words import (
        Edges,
        Labels,
    )

    rows = get_neo4j_conn().query_read(
        f"""
        MATCH (db:{Labels.DB} {{name: $db}})-[:{Edges.CONTAINS}]->
              (sch:{Labels.SCHEMA})-[:{Edges.CONTAINS}]->(t:{Labels.TABLE})
              -[:{Edges.CONTAINS}]->(c:{Labels.COLUMN})
        WHERE toLower(c.data_type) CONTAINS 'json'
        RETURN DISTINCT t.id AS id, t.name AS name, sch.name AS schema_name
        ORDER BY t.name
        """,
        {"db": database_name},
    )
    return rows


def _clear_jsonb_sample_values(database_name: str) -> int:
    """NULL out sample_values on all JSONB Column nodes for this database.

    Required before re-profiling when a previous compilation stored raw JSON
    strings: store_column_sample_values uses coalesce (set-if-null), so it
    cannot overwrite an existing non-null value without this step.
    Returns number of columns cleared.
    """
    from nemo_retriever.tabular_data.ingestion.model.reserved_words import (
        Edges,
        Labels,
    )

    result = get_neo4j_conn().query_write(
        f"""
        MATCH (db:{Labels.DB} {{name: $db}})-[:{Edges.CONTAINS}]->
              (:{Labels.SCHEMA})-[:{Edges.CONTAINS}]->(:{Labels.TABLE})
              -[:{Edges.CONTAINS}]->(c:{Labels.COLUMN})
        WHERE toLower(c.data_type) CONTAINS 'json'
          AND c.sample_values IS NOT NULL
        SET c.sample_values = null
        RETURN count(c) AS cleared
        """,
        {"db": database_name},
    )
    return result[0]["cleared"] if result else 0


def reprofiling_for_connection(connection_string: str, *, force: bool = False) -> int:
    """Re-profile JSONB tables for one source database. Returns count updated.

    When *force* is True, any existing sample_values on JSONB columns are
    cleared first so the profiling can overwrite them (bypasses the coalesce
    set-if-null guard in store_column_sample_values).
    """
    import gsf.dal.datasources as _ds
    import gsf.semantic.visit_enter as _ve

    connector = create_connector(connection_string)
    db_name = connector.database_name
    logger.info("=== %s ===", db_name)

    if force:
        cleared = _clear_jsonb_sample_values(db_name)
        logger.info("  --force: cleared sample_values on %d JSONB column(s)", cleared)

    tables = _find_jsonb_tables_in_neo4j(db_name)
    if not tables:
        logger.info("  No JSONB tables found in Neo4j for %r — skipping", db_name)
        return 0

    logger.info("  Found %d table(s) with JSONB columns", len(tables))
    updated = 0
    failed = 0

    for table in tables:
        table_id = table["id"]
        table_name = table["name"]
        ctx = fetch_table_context(table_id)
        columns = ctx.get("columns", [])

        if not columns:
            logger.warning("  [%s] no columns in Neo4j — skipping", table_name)
            continue

        jsonb_col_names = [
            c["name"] for c in columns if "json" in (c.get("data_type") or "").lower()
        ]

        # Intercept what store_column_sample_values actually receives so we log
        # the stored keys rather than the raw profiling data. Must patch the
        # name as bound inside visit_enter.py (via its own `from ... import`),
        # not the gsf.dal.datasources module attribute — calculate_columns_
        # profiling calls its own already-bound reference, so patching the
        # datasources module attribute alone is silently never observed by it
        # (the real write still goes through, only this capture is skipped).
        stored_for_table: dict[str, list] = {}
        _orig_store = _ve.store_column_sample_values

        def _capturing_store(tid: str, samples: dict, _orig=_orig_store) -> None:
            stored_for_table.update(samples)
            _orig(tid, samples)

        _ds.store_column_sample_values = _capturing_store
        _ve.store_column_sample_values = _capturing_store
        logger.info("  Re-profiling %s ...", table_name)
        try:
            calculate_columns_profiling(table, columns, connector)
            updated += 1
        except Exception:
            logger.exception("  [%s] profiling failed", table_name)
            failed += 1
        finally:
            _ds.store_column_sample_values = _orig_store
            _ve.store_column_sample_values = _orig_store

        for col_name in jsonb_col_names:
            stored = stored_for_table.get(col_name, [])
            if stored:
                logger.info("    %-30s → %s", col_name, stored[:5])
            else:
                logger.info("    %-30s → (no keys stored)", col_name)

    logger.info(
        "  Updated %d/%d table(s)%s",
        updated,
        len(tables),
        f" ({failed} failed)" if failed else "",
    )
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--connection",
        "-c",
        metavar="CONN",
        action="append",
        default=[],
        help="Connection string to re-profile (repeatable)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Clear existing JSONB sample_values in Neo4j before profiling. "
            "Required when a prior compilation stored raw JSON strings that "
            "the coalesce guard would otherwise preserve."
        ),
    )
    args = parser.parse_args()

    connections: list[str] = list(args.connection)

    if not connections:
        parser.error("Specify at least one --connection")

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for c in connections:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    total = 0
    for conn in unique:
        try:
            total += reprofiling_for_connection(conn, force=args.force)
        except Exception:
            logger.exception("Failed for connection %r", conn)

    logger.info(
        "Done — %d table(s) re-profiled across %d database(s)", total, len(unique)
    )


if __name__ == "__main__":
    main()
