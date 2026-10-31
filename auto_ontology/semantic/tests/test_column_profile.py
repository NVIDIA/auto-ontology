# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sqlite3

import pandas as pd
import pytest

from auto_ontology.retrieval.text_to_sql import formatters_util
from auto_ontology.semantic.column_profile import (
    _resolve_names,
    measure_column_profiles,
    value_shape,
)
from auto_ontology.utils.column_profile import (
    dump_profile,
    parse_profile,
    profile_facts,
    profile_text,
)


class _SqliteConnector:
    dialect = "sqlite"
    supports_statement_timeout = False

    def __init__(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        self.statements: list[str] = []

    def qualify(self, schema, name):
        return f'"{name}"'

    def execute(self, sql):
        self.statements.append(sql)
        return pd.read_sql_query(sql, self.conn)


@pytest.fixture
def connector():
    c = _SqliteConnector()
    c.conn.executescript(
        """
        CREATE TABLE schools (code TEXT, county TEXT, kind TEXT, score REAL, blob BLOB);
        INSERT INTO schools VALUES
            ('0101', 'Alameda', 'A', 1.5, NULL),
            ('0102', 'Alameda', '', 2.5, NULL),
            ('0103', 'Yuba', 'A', NULL, NULL),
            ('0104', NULL, 'B', 4.0, NULL);
        """
    )
    return c


COLUMNS = [
    {"name": "code", "data_type": "TEXT"},
    {"name": "county", "data_type": "TEXT"},
    {"name": "kind", "data_type": "TEXT"},
    {"name": "score", "data_type": "REAL"},
    {"name": "blob", "data_type": "BLOB"},
]


def test_measure_reads_the_whole_table(connector):
    profiles = measure_column_profiles(connector, {"name": "schools"}, COLUMNS)

    assert set(profiles) == {"code", "county", "kind", "score"}
    code = profiles["code"]
    assert code["row_count"] == 4
    assert code["null_count"] == 0
    assert code["distinct_count"] == 4
    assert (code["min"], code["max"]) == ("0101", "0104")
    assert code["min_length"] == code["max_length"] == 4
    assert "top_values" not in code
    assert code["shape"][0] == "digits only"

    county = profiles["county"]
    assert county["null_count"] == 1
    assert county["top_values"][0] == ["Alameda", 2]

    assert profiles["kind"]["empty_count"] == 1
    assert "min_length" not in profiles["score"]


def test_one_unaggregable_column_keeps_the_rest(connector, monkeypatch):
    execute = connector.execute

    def refuse_county(sql):
        if '"county"' in sql:
            raise RuntimeError("cannot aggregate")
        return execute(sql)

    monkeypatch.setattr(connector, "execute", refuse_county)

    profiles = measure_column_profiles(connector, {"name": "schools"}, COLUMNS[:3])

    assert set(profiles) == {"code", "kind"}


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["01", "02"], ["digits only"]),
        (["1.5", "-2"], ["numbers stored as text"]),
        (["AB", "CD1"], ["upper case"]),
        (["ab ", "cd"], ["lower case", "some values have leading or trailing spaces"]),
        (["ID-1", "ID-2"], ["upper case", "every value starts with 'ID-'"]),
        ([1, 2], []),
    ],
)
def test_value_shape(values, expected):
    assert value_shape(values) == expected


def test_profile_round_trip():
    profile = {"row_count": 3, "summary": "x"}
    assert parse_profile(dump_profile(profile)) == profile
    assert dump_profile({}) is None
    assert parse_profile("not json") is None
    assert parse_profile(None) is None


def test_profile_facts_wording():
    profile = {
        "row_count": 1000,
        "null_count": 3,
        "distinct_count": 4,
        "min": "A",
        "max": "D",
        "min_length": 1,
        "max_length": 1,
        "empty_count": 0,
        "shape": ["upper case"],
        "top_values": [["A", 600], ["B", 300], ["Z", 1]],
    }

    facts = profile_facts(profile)
    assert facts == (
        "3 of 1,000 rows NULL (under 1%); 4 distinct values; from 'A' to 'D'; "
        "always 1 characters; upper case; 'A' in 60% of non-NULL rows"
    )
    assert profile_facts(profile, top_values=True).endswith(
        "most common: 'A' (600), 'B' (300)"
    )


def test_profile_text_puts_summary_first():
    profile = {"row_count": 2, "null_count": 0, "summary": "Two-letter state code."}
    assert profile_text(profile) == "Two-letter state code. (no NULLs)"
    assert profile_text(None) == ""


def test_resolve_names_accepts_echoed_type():
    assert _resolve_names("code (TEXT)", ["co", "code"]) == "code"
    assert _resolve_names("other", ["code"]) is None


def test_prompt_shows_profile_unless_switched_off(monkeypatch):
    table = {
        "name": "schools",
        "columns": [
            {
                "name": "code",
                "data_type": "TEXT",
                "profile": {"row_count": 2, "null_count": 0, "summary": "S|code"},
            }
        ],
    }

    monkeypatch.delenv("SQL_COLUMN_PROFILE", raising=False)
    assert "S/code (no NULLs)" in formatters_util.format_tables_for_prompt([table])

    monkeypatch.setenv("SQL_COLUMN_PROFILE", "false")
    assert "profile" not in formatters_util.format_tables_for_prompt([table])
