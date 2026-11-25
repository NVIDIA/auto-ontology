# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The join-fanout check: which aggregates a join repeats, and what submit does.

The fixture has three suppliers with 1, 2 and 19 ASNs. Every query is run on
SQLite too, and compared with an oracle computed from the rows in Python, so a
query the check refuses is shown to be wrong and one it passes to be right.
Key profiles come from the same data, standing in for catalog profiling.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable

import pytest
import sqlglot
from langchain_core.messages import AIMessage

from auto_ontology.retrieval.text_to_sql.react import fanout, loop
from auto_ontology.retrieval.text_to_sql.react import run_sql as run_sql_module
from auto_ontology.retrieval.text_to_sql.react.context import ToolContext
from auto_ontology.retrieval.text_to_sql.react.fanout import KeyProfile
from auto_ontology.retrieval.text_to_sql.tests.react_fakes import (
    FakeCatalog,
    ScriptedModel,
    SQLiteConnector,
    call,
)

ASNS_PER_SUPPLIER = {1: 1, 2: 2, 3: 19}
RATES = {1: 10.0, 2: 20.0, 3: 30.0}


def _script() -> str:
    lines = [
        "CREATE TABLE suppliers (id INTEGER PRIMARY KEY, name TEXT, rate REAL);",
        "CREATE TABLE asn (id INTEGER PRIMARY KEY, supplier_id INTEGER, qty REAL);",
        "CREATE TABLE sourcing (id INTEGER PRIMARY KEY, supplier_id INTEGER, "
        "share REAL);",
        "CREATE TABLE contracts (supplier_id INTEGER, terms TEXT);",
    ]
    asn_id = 0
    for supplier, count in ASNS_PER_SUPPLIER.items():
        lines.append(
            f"INSERT INTO suppliers VALUES ({supplier}, 's{supplier}', "
            f"{RATES[supplier]});"
        )
        lines.append(f"INSERT INTO contracts VALUES ({supplier}, 'net30');")
        for _ in range(count):
            asn_id += 1
            lines.append(f"INSERT INTO asn VALUES ({asn_id}, {supplier}, {asn_id});")
    # Two sourcing rows per supplier, so sourcing repeats asn and the reverse.
    for supplier in ASNS_PER_SUPPLIER:
        for part in range(2):
            row = supplier * 10 + part
            lines.append(f"INSERT INTO sourcing VALUES ({row}, {supplier}, 50.0);")
    return "\n".join(lines)


SCRIPT = _script()
TABLES = {
    "suppliers": ["id", "name", "rate"],
    "asn": ["id", "supplier_id", "qty"],
    "sourcing": ["id", "supplier_id", "share"],
    "contracts": ["supplier_id", "terms"],
}
CATALOG = FakeCatalog(TABLES)


def _profiles_from(conn: sqlite3.Connection) -> dict[str, KeyProfile]:
    """What profiling would store: the declared key, and each column's ratio."""
    profiles = {}
    for table, columns in TABLES.items():
        info = conn.execute(f"PRAGMA table_info({table})").fetchall()
        pk = [row[1] for row in info if row[5]]
        stats = {}
        for column in columns:
            distinct, total = conn.execute(
                f"SELECT COUNT(DISTINCT {column}), COUNT({column}) FROM {table}"
            ).fetchone()
            ratio = distinct / total if total else None
            stats[column] = (ratio is not None and ratio >= 0.95, ratio)
        profiles[table] = KeyProfile.from_catalog({"pk": pk, "columns": stats})
    return profiles


@pytest.fixture(scope="module")
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCRIPT)
    return conn


@pytest.fixture(scope="module")
def profiles(db: sqlite3.Connection) -> dict[str, KeyProfile]:
    return _profiles_from(db)


def _fanouts(sql: str, profiles: dict[str, KeyProfile]) -> list[fanout.Fanout]:
    statement = sqlglot.parse_one(sql, read="sqlite")
    return fanout.find_fanouts(
        statement, CATALOG, lambda table, _schema: profiles.get(table.lower())
    )


def _qty_total(supplier: int) -> float:
    first = 1 + sum(c for s, c in ASNS_PER_SUPPLIER.items() if s < supplier)
    return float(sum(range(first, first + ASNS_PER_SUPPLIER[supplier])))


def _rows(db: sqlite3.Connection, sql: str) -> dict:
    return {row[0]: tuple(row[1:]) for row in db.execute(sql).fetchall()}


# -- the profile -------------------------------------------------------------


def test_profile_reads_declared_keys_and_measured_repeats(
    profiles: dict[str, KeyProfile],
) -> None:
    assert profiles["suppliers"].unique >= {"id"}
    assert "supplier_id" in profiles["asn"].repeats
    assert "supplier_id" in profiles["contracts"].unique


def test_profile_without_a_measured_ratio_proves_nothing() -> None:
    profile = KeyProfile.from_catalog(
        {"pk": None, "columns": {"supplier_id": (False, None)}}
    )

    assert profile.repeats == frozenset()
    assert profile.unique == frozenset()


def test_profile_mostly_distinct_still_repeats() -> None:
    """Flagged unique at a 0.97 ratio still saw a value twice."""
    profile = KeyProfile.from_catalog({"pk": [], "columns": {"code": (True, 0.97)}})

    assert profile.repeats == {"code"}
    assert profile.unique == frozenset()


def test_profile_declared_single_column_key_is_unique() -> None:
    profile = KeyProfile.from_catalog({"pk": ["ID"], "columns": {}})

    assert profile.unique == {"id"}


def test_profile_composite_key_member_is_not_unique() -> None:
    profile = KeyProfile.from_catalog(
        {"pk": ["order_id", "line"], "columns": {"order_id": (False, 0.5)}}
    )

    assert profile.unique == frozenset()
    assert profile.repeats == {"order_id"}


# -- refused: an aggregate the join repeats ------------------------------------


GROUPED_OWNER_SUM = (
    "SELECT s.id, SUM(s.rate) FROM suppliers s "
    "JOIN asn a ON a.supplier_id = s.id GROUP BY s.id"
)


def test_grouped_owner_sum_after_fanout_is_refused_and_wrong(
    db: sqlite3.Connection, profiles: dict[str, KeyProfile]
) -> None:
    found = _fanouts(GROUPED_OWNER_SUM, profiles)

    assert [(f.aggregate, f.detail, f.key) for f in found] == [
        ("SUM(s.rate)", "asn", "supplier_id")
    ]
    rows = _rows(db, GROUPED_OWNER_SUM)
    for supplier, count in ASNS_PER_SUPPLIER.items():
        assert rows[supplier] == (RATES[supplier] * count,)
    # Right only for the supplier with one ASN, which a catalog check cannot see.
    assert rows[1] == (RATES[1],)
    assert rows[2] != (RATES[2],)
    assert rows[3] != (RATES[3],)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT SUM(s.rate) FROM suppliers s JOIN asn a ON a.supplier_id = s.id",
        "SELECT AVG(s.rate) FROM suppliers s JOIN asn a ON a.supplier_id = s.id",
        "SELECT COUNT(s.id) FROM suppliers s JOIN asn a ON a.supplier_id = s.id",
        "SELECT SUM(s.rate) FROM asn a JOIN suppliers s ON s.id = a.supplier_id",
        "SELECT SUM(s.rate) FROM suppliers s, asn a WHERE a.supplier_id = s.id",
        "SELECT SUM(s.rate) FROM suppliers s LEFT JOIN asn a ON s.id = a.supplier_id",
        "SELECT SUM(rate) FROM suppliers JOIN asn ON asn.supplier_id = suppliers.id",
        "SELECT SUM(s.rate) FROM suppliers s "
        "JOIN asn a ON CAST(a.supplier_id AS INTEGER) = s.id",
        "SELECT s.name FROM suppliers s JOIN asn a ON a.supplier_id = s.id "
        "GROUP BY s.name HAVING SUM(s.rate) > 15",
        "WITH t AS (SELECT s.id, SUM(s.rate) AS r FROM suppliers s "
        "JOIN asn a ON a.supplier_id = s.id GROUP BY s.id) SELECT * FROM t",
        "SELECT x.r FROM (SELECT SUM(s.rate) AS r FROM suppliers s "
        "JOIN asn a ON a.supplier_id = s.id) x",
    ],
)
def test_owner_measure_after_fanout_is_refused(
    db: sqlite3.Connection, profiles: dict[str, KeyProfile], sql: str
) -> None:
    found = _fanouts(sql, profiles)

    assert [(f.detail, f.key) for f in found] == [("asn", "supplier_id")]


def test_ungrouped_owner_sum_is_inflated(db: sqlite3.Connection) -> None:
    (total,) = db.execute(
        "SELECT SUM(s.rate) FROM suppliers s JOIN asn a ON a.supplier_id = s.id"
    ).fetchone()

    assert total != sum(RATES.values())
    assert total == sum(RATES[s] * c for s, c in ASNS_PER_SUPPLIER.items())


def test_two_detail_tables_repeat_each_other(
    db: sqlite3.Connection, profiles: dict[str, KeyProfile]
) -> None:
    """The issue's shape: each detail's total is multiplied by the other's rows."""
    sql = (
        "SELECT s.id, SUM(sr.share), SUM(a.qty) FROM suppliers s "
        "LEFT JOIN sourcing sr ON sr.supplier_id = s.id "
        "LEFT JOIN asn a ON a.supplier_id = s.id GROUP BY s.id"
    )

    found = _fanouts(sql, profiles)

    assert sorted((f.aggregate, f.detail) for f in found) == [
        ("SUM(a.qty)", "sourcing"),
        ("SUM(sr.share)", "asn"),
    ]
    rows = _rows(db, sql)
    assert rows[3] == (100.0 * 19, _qty_total(3) * 2)


def test_using_reads_only_the_sources_left_of_the_join(
    profiles: dict[str, KeyProfile],
) -> None:
    """A later join whose table has the same column does not make USING unclear."""
    sql = (
        "SELECT SUM(sr.share) FROM sourcing sr JOIN asn a USING (supplier_id) "
        "JOIN contracts c ON c.supplier_id = sr.supplier_id"
    )

    assert [(f.detail, f.key) for f in _fanouts(sql, profiles)] == [
        ("asn", "supplier_id")
    ]


def test_aggregate_filter_spares_only_that_aggregate(
    profiles: dict[str, KeyProfile],
) -> None:
    sql = (
        "SELECT SUM(s.rate) FILTER (WHERE a.id > 2), SUM(s.rate) FROM suppliers s "
        "JOIN asn a ON a.supplier_id = s.id"
    )

    assert [f.aggregate for f in _fanouts(sql, profiles)] == ["SUM(s.rate)"]


def test_grouping_by_a_cast_key_does_not_spare_it(
    profiles: dict[str, KeyProfile],
) -> None:
    sql = (
        "SELECT SUM(s.rate) FROM suppliers s JOIN asn a ON a.supplier_id = s.id "
        "GROUP BY CAST(a.id AS VARCHAR(1))"
    )

    assert [f.detail for f in _fanouts(sql, profiles)] == ["asn"]


def test_join_using_is_read(profiles: dict[str, KeyProfile]) -> None:
    sql = "SELECT SUM(sr.share) FROM sourcing sr JOIN asn a USING (supplier_id)"

    assert [(f.detail, f.key) for f in _fanouts(sql, profiles)] == [
        ("asn", "supplier_id")
    ]


def test_message_says_how_to_fix_it(profiles: dict[str, KeyProfile]) -> None:
    (found,) = _fanouts(GROUPED_OWNER_SUM, profiles)

    text = found.describe()
    assert "SUM(s.rate) reads suppliers rows" in text
    assert "a.supplier_id" in text
    assert "grouped by supplier_id before joining it" in text
    assert "count the distinct values of its key" in text


# -- accepted: owner-grain and pre-aggregated queries ------------------------


def _rate(supplier: int) -> tuple:
    return (RATES[supplier],)


def _rate_and_qty(supplier: int) -> tuple:
    return (RATES[supplier], _qty_total(supplier))


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        # The owner's value, projected at its own grain.
        ("SELECT s.id, s.rate FROM suppliers s", _rate),
        # Detail pre-aggregated in a CTE, then joined one row per owner.
        (
            "WITH q AS (SELECT supplier_id, SUM(qty) AS qty FROM asn "
            "GROUP BY supplier_id) "
            "SELECT s.id, s.rate, q.qty FROM suppliers s "
            "JOIN q ON q.supplier_id = s.id",
            _rate_and_qty,
        ),
        # The same in a subquery, summed again outside.
        (
            "SELECT s.id, SUM(s.rate), SUM(q.qty) FROM suppliers s JOIN "
            "(SELECT supplier_id, SUM(qty) AS qty FROM asn GROUP BY supplier_id) q "
            "ON q.supplier_id = s.id GROUP BY s.id",
            _rate_and_qty,
        ),
        # Each detail aggregated on its own, then joined at the owner grain.
        (
            "WITH q AS (SELECT supplier_id, SUM(qty) AS qty FROM asn "
            "GROUP BY supplier_id), "
            "r AS (SELECT supplier_id, SUM(share) AS share FROM sourcing "
            "GROUP BY supplier_id) "
            "SELECT s.id, SUM(s.rate), SUM(q.qty), SUM(r.share) FROM suppliers s "
            "JOIN q ON q.supplier_id = s.id JOIN r ON r.supplier_id = s.id "
            "GROUP BY s.id",
            lambda supplier: (*_rate_and_qty(supplier), 100.0),
        ),
        # A detail measure through a join on the owner's unique key.
        (
            "SELECT s.id, s.rate, SUM(a.qty) FROM asn a "
            "JOIN suppliers s ON s.id = a.supplier_id GROUP BY s.id, s.rate",
            _rate_and_qty,
        ),
        # A one-to-one join.
        (
            "SELECT s.id, SUM(s.rate) FROM suppliers s "
            "JOIN contracts c ON c.supplier_id = s.id GROUP BY s.id",
            _rate,
        ),
    ],
)
def test_owner_grain_and_pre_aggregation_pass_with_the_oracle_values(
    db: sqlite3.Connection,
    profiles: dict[str, KeyProfile],
    sql: str,
    expected: Callable[[int], tuple],
) -> None:
    assert _fanouts(sql, profiles) == []
    rows = _rows(db, sql)
    assert rows == {supplier: expected(supplier) for supplier in ASNS_PER_SUPPLIER}


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT COUNT(DISTINCT s.id) FROM suppliers s "
        "JOIN asn a ON a.supplier_id = s.id",
        "SELECT SUM(DISTINCT s.rate) FROM suppliers s "
        "JOIN asn a ON a.supplier_id = s.id",
        "SELECT COUNT(*) FROM suppliers s JOIN asn a ON a.supplier_id = s.id",
        "SELECT COUNT(a.id) FROM suppliers s JOIN asn a ON a.supplier_id = s.id",
        "SELECT MAX(s.rate) FROM suppliers s JOIN asn a ON a.supplier_id = s.id",
        # Detail and owner measure together: the detail row is the grain.
        "SELECT SUM(a.qty * s.rate) FROM suppliers s "
        "JOIN asn a ON a.supplier_id = s.id",
        # A window does not collapse rows.
        "SELECT a.id, SUM(s.rate) OVER () FROM suppliers s "
        "JOIN asn a ON a.supplier_id = s.id",
        # Grouped by the detail's unique key: one detail row per group.
        "SELECT a.id, SUM(s.rate) FROM suppliers s "
        "JOIN asn a ON a.supplier_id = s.id GROUP BY a.id",
        "SELECT a.id, SUM(s.rate) FROM suppliers s "
        "JOIN asn a ON a.supplier_id = s.id GROUP BY 1",
        # An equality filter may leave one detail row per owner.
        "SELECT SUM(s.rate) FROM suppliers s JOIN asn a ON a.supplier_id = s.id "
        "AND a.qty = 3",
        "SELECT SUM(s.rate) FROM suppliers s JOIN asn a ON a.supplier_id = s.id "
        "WHERE a.id IN (1, 2)",
        "SELECT SUM(s.rate) FILTER (WHERE a.id = 2) FROM suppliers s "
        "JOIN asn a ON a.supplier_id = s.id",
        # Any condition on the joined table may keep one row per key: a
        # degenerate range, or a validity window.
        "SELECT SUM(s.rate) FROM suppliers s JOIN asn a ON a.supplier_id = s.id "
        "WHERE a.id BETWEEN 2 AND 2",
        "SELECT SUM(s.rate) FROM suppliers s JOIN asn a ON a.supplier_id = s.id "
        "AND a.qty <= 3 AND a.id > 2",
        "SELECT SUM(s.rate) FILTER (WHERE a.id > 2) FROM suppliers s "
        "JOIN asn a ON a.supplier_id = s.id",
        # A correlated subquery that picks the latest row per owner.
        "SELECT SUM(s.rate) FROM suppliers s JOIN asn a ON a.supplier_id = s.id "
        "WHERE a.id = (SELECT MAX(a2.id) FROM asn a2 "
        "WHERE a2.supplier_id = a.supplier_id)",
        # A join on two columns: the pair may be unique.
        "SELECT SUM(s.rate) FROM suppliers s "
        "JOIN asn a ON a.supplier_id = s.id AND a.qty = s.rate",
        # Semi-joins do not repeat rows.
        "SELECT SUM(s.rate) FROM suppliers s "
        "WHERE s.id IN (SELECT supplier_id FROM asn)",
        # Correlated scalar subquery at the owner grain.
        "SELECT s.id, (SELECT SUM(a.qty) FROM asn a "
        "WHERE a.supplier_id = s.id) FROM suppliers s",
    ],
)
def test_shapes_that_do_not_repeat_rows_pass(
    profiles: dict[str, KeyProfile], sql: str
) -> None:
    assert _fanouts(sql, profiles) == []


def test_grouped_by_detail_key_keeps_the_owner_value(db: sqlite3.Connection) -> None:
    rows = _rows(
        db,
        "SELECT a.id, SUM(s.rate), s.id FROM suppliers s "
        "JOIN asn a ON a.supplier_id = s.id GROUP BY a.id",
    )

    assert all(rate == RATES[supplier] for rate, supplier in rows.values())


# -- unknown uniqueness passes -----------------------------------------------


def test_no_profile_passes() -> None:
    assert _fanouts(GROUPED_OWNER_SUM, {}) == []


def test_unmeasured_join_column_passes(profiles: dict[str, KeyProfile]) -> None:
    unmeasured = dict(profiles)
    unmeasured["asn"] = KeyProfile.from_catalog(
        {"pk": ["id"], "columns": {"supplier_id": (False, None)}}
    )

    assert _fanouts(GROUPED_OWNER_SUM, unmeasured) == []


def test_unique_join_column_passes(profiles: dict[str, KeyProfile]) -> None:
    unique = dict(profiles)
    unique["asn"] = KeyProfile.from_catalog(
        {"pk": ["id"], "columns": {"supplier_id": (True, 1.0)}}
    )

    assert _fanouts(GROUPED_OWNER_SUM, unique) == []


def test_ambiguous_unqualified_column_passes(profiles: dict[str, KeyProfile]) -> None:
    sql = "SELECT SUM(id) FROM suppliers s JOIN asn a ON a.supplier_id = s.id"

    assert _fanouts(sql, profiles) == []


def test_failing_profile_lookup_finds_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*_args: object) -> None:
        raise RuntimeError("store down")

    monkeypatch.setattr(fanout, "key_profiles", lambda *_: broken)
    statement = sqlglot.parse_one(GROUPED_OWNER_SUM, read="sqlite")

    ctx = ToolContext(retriever=None, connectors=[])
    assert fanout.safe_find_fanouts(statement, CATALOG, ctx, None) == []


# -- submit and run_sql ------------------------------------------------------


@pytest.fixture
def ctx(
    monkeypatch: pytest.MonkeyPatch, profiles: dict[str, KeyProfile]
) -> ToolContext:
    monkeypatch.setattr(run_sql_module, "catalog_for", lambda *_: CATALOG)
    monkeypatch.setattr(
        fanout,
        "key_profiles",
        lambda *_: lambda table, _schema: profiles.get(table.lower()),
    )
    context = ToolContext(retriever=None, connectors=[SQLiteConnector(SCRIPT)])
    for name in TABLES:
        context.show_table(None, name)
    return context


PRE_AGGREGATED = (
    "SELECT s.id, s.rate FROM suppliers s JOIN "
    "(SELECT supplier_id FROM asn GROUP BY supplier_id) q ON q.supplier_id = s.id"
)


def test_submit_refuses_once_then_accepts_the_same_sql(ctx: ToolContext) -> None:
    observation, submission = loop.submit(ctx, GROUPED_OWNER_SUM, "x")

    assert submission is None
    assert observation.startswith("Not accepted. A join repeats")
    assert "submit this version again unchanged" in observation

    observation, submission = loop.submit(ctx, GROUPED_OWNER_SUM, "x")

    assert observation == "Accepted."
    assert submission is not None


def test_refusal_says_how_to_resend_through_run_sql(ctx: ToolContext) -> None:
    resubmit = "run this version with final=true"

    observation, _ = loop.submit(ctx, GROUPED_OWNER_SUM, "x", resubmit=resubmit)

    assert observation.endswith(f"{resubmit} again unchanged.")


def test_submit_accepts_a_fixed_query_first_time(ctx: ToolContext) -> None:
    assert loop.submit(ctx, GROUPED_OWNER_SUM, "x")[1] is None

    observation, submission = loop.submit(ctx, PRE_AGGREGATED, "x")

    assert observation == "Accepted."
    assert submission is not None


def test_run_sql_runs_and_adds_a_note(ctx: ToolContext) -> None:
    text = run_sql_module.run_sql(ctx, GROUPED_OWNER_SUM)

    assert text.startswith("3 rows")
    assert "Note: a join repeats the rows an aggregate adds up" in text
    assert "Note:" not in run_sql_module.run_sql(ctx, PRE_AGGREGATED)
    # A note is not a refusal: submit still refuses the query once.
    assert loop.submit(ctx, GROUPED_OWNER_SUM, "x")[1] is None


def test_run_sql_note_fits_the_observation_limit(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rows are clipped to leave room for the note, not the note dropped."""
    monkeypatch.setenv("TEXT_TO_SQL_REACT_MAX_OBSERVATION_CHARS", "400")
    sql = (
        "SELECT a.qty % 11 AS bucket, SUM(s.rate), MAX(s.name) FROM suppliers s "
        "JOIN asn a ON a.supplier_id = s.id GROUP BY a.qty % 11"
    )

    text = run_sql_module.run_sql(ctx, sql)

    assert len(text) == 400
    assert text.startswith("11 rows")
    assert "…\n\nNote: a join repeats" in text


def test_two_findings_stay_within_a_small_limit(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEXT_TO_SQL_REACT_MAX_OBSERVATION_CHARS", "600")
    sql = (
        "SELECT s.id, SUM(sr.share), SUM(a.qty) FROM suppliers s "
        "LEFT JOIN sourcing sr ON sr.supplier_id = s.id "
        "LEFT JOIN asn a ON a.supplier_id = s.id GROUP BY s.id"
    )

    text = run_sql_module.run_sql(ctx, sql)

    assert len(text) <= 600
    assert text.startswith("3 rows")
    assert "Note: a join repeats" in text


def test_a_result_clipped_for_the_note_is_saved_for_grep(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rows that fit alone but not beside the note go to a page, not nowhere."""
    monkeypatch.setenv("TEXT_TO_SQL_REACT_GREP", "true")
    monkeypatch.setenv("TEXT_TO_SQL_REACT_MAX_OBSERVATION_CHARS", "400")
    sql = (
        "SELECT a.qty % 11 AS bucket, SUM(s.rate), MAX(s.name) FROM suppliers s "
        "JOIN asn a ON a.supplier_id = s.id GROUP BY a.qty % 11"
    )

    text = run_sql_module.run_sql(ctx, sql)

    assert len(text) <= 400
    assert "Saved as page 1" in text.splitlines()[0]
    assert len(ctx.pages[0]) == 11


def test_episode_fixes_the_query_after_the_refusal(ctx: ToolContext) -> None:
    model = ScriptedModel(
        [
            AIMessage(
                content="",
                tool_calls=[call("submit", "1", sql=GROUPED_OWNER_SUM, response="x")],
            ),
            AIMessage(
                content="",
                tool_calls=[call("submit", "2", sql=PRE_AGGREGATED, response="x")],
            ),
        ]
    )

    result = loop.run_episode(model, ctx, "What is each supplier's rate?")

    assert result.reason == "submitted"
    assert result.submission.sql == PRE_AGGREGATED
    refusal = [t for t in result.trajectory if t["role"] == "tool"][0]["content"]
    assert "asn" in refusal
