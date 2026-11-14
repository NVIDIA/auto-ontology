# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from auto_ontology.connectors.registry import parse_connection_strings


def test_parse_connection_strings_accepts_commas_and_newlines() -> None:
    raw = """
        postgresql://localhost/first
        clickhouse://localhost/second,

        snowflake://account?database=third
    """

    assert parse_connection_strings(raw) == [
        "postgresql://localhost/first",
        "clickhouse://localhost/second",
        "snowflake://account?database=third",
    ]


def test_commas_inside_a_url_do_not_split_it() -> None:
    raw = (
        "mongodb://mongodb:27017,mongo-2:27017/shop?replicaSet=rs0"
        "&readPreferenceTags=dc:ny,rack:r1,"
        "postgresql://localhost/first\n"
        "sqlite:///tmp/x.sqlite, duckdb:///tmp/y.duckdb,"
        "mongodb+srv://cluster0.example.net/shop"
    )
    assert parse_connection_strings(raw) == [
        "mongodb://mongodb:27017,mongo-2:27017/shop?replicaSet=rs0"
        "&readPreferenceTags=dc:ny,rack:r1",
        "postgresql://localhost/first",
        "sqlite:///tmp/x.sqlite",
        "duckdb:///tmp/y.duckdb",
        "mongodb+srv://cluster0.example.net/shop",
    ]


def test_a_url_without_slashes_still_starts_a_new_connection_string() -> None:
    assert parse_connection_strings("postgresql://h/db1,SQLite:data/app.sqlite") == [
        "postgresql://h/db1",
        "SQLite:data/app.sqlite",
    ]
