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
