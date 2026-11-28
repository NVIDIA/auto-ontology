# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Declared DuckDB constraints must survive the connector/catalog boundary."""

from pathlib import Path

import duckdb
import pandas as pd

from auto_ontology.catalog.extract import extract_tabular_db_data
from auto_ontology.connectors.duckdb import DuckDBDatabase


def test_declared_keys_are_extracted_read_only(tmp_path: Path) -> None:
    path = tmp_path / "keys.duckdb"
    with duckdb.connect(str(path)) as conn:
        conn.execute('CREATE SCHEMA "model schema"')
        conn.execute(
            'CREATE TABLE "model schema".sites '
            '(region VARCHAR, "Site ID" VARCHAR, PRIMARY KEY ("Site ID", region))'
        )
        conn.execute(
            'CREATE TABLE "model schema".history '
            "(area VARCHAR, site VARCHAR, period INTEGER, "
            "PRIMARY KEY (site, period), FOREIGN KEY (site, area) "
            'REFERENCES "model schema".sites ("Site ID", region))'
        )
        conn.execute(
            'CREATE VIEW "model schema".history_view AS '
            'SELECT * FROM "model schema".history'
        )
    with DuckDBDatabase(str(path)) as db:
        data = extract_tabular_db_data(db)
        assert data["pks"].to_dict("records") == [
            {
                "table_schema": "model schema",
                "table_name": table,
                "column_name": column,
                "ordinal_position": position,
            }
            for table, column, position in [
                ("history", "site", 1),
                ("history", "period", 2),
                ("sites", "Site ID", 1),
                ("sites", "region", 2),
            ]
        ]
        assert data["fks"].to_dict("records") == [
            {
                "table_schema": "model schema",
                "table_name": "history",
                "column_name": column,
                "referenced_schema": "model schema",
                "referenced_table": "sites",
                "referenced_column": target,
            }
            for column, target in [("site", "Site ID"), ("area", "region")]
        ]


def test_constraint_free_database_returns_shaped_empty_frames(tmp_path: Path) -> None:
    path = tmp_path / "no_keys.duckdb"
    with duckdb.connect(str(path)) as conn:
        conn.execute("CREATE TABLE events (id INTEGER, site_id INTEGER)")
        conn.execute("INSERT INTO events VALUES (1, 1), (2, 1)")
    with DuckDBDatabase(str(path)) as db:
        pd.testing.assert_frame_equal(
            db.get_pks(),
            pd.DataFrame(
                columns=[
                    "table_schema",
                    "table_name",
                    "column_name",
                    "ordinal_position",
                ]
            ).astype({"ordinal_position": "int64"}),
        )
        assert db.get_fks().empty
        assert list(db.get_fks().columns) == [
            "table_schema",
            "table_name",
            "column_name",
            "referenced_schema",
            "referenced_table",
            "referenced_column",
        ]
