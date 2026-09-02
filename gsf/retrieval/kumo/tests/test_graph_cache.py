# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusing a built graph, and the cases where reuse would be wrong.

A follow-up about the same tables should not read them again. A question against
a changed schema, a different prompt, or a different model must not be answered
from what an earlier one built.
"""

import threading
import time

import pytest
from pathlib import Path
from typing import Any

from gsf.retrieval.kumo.graph_cache import (
    BuildTimedOut,
    CacheKey,
    GraphCache,
    catalog_fingerprint,
)


def _key(**overrides) -> CacheKey:
    fields = {
        "connector": "SQLiteDatabase:sales:sqlite",
        "database": "sales",
        "tables": ("customers", "orders"),
        "schema_fingerprint": "abc123",
        "join_fingerprint": "j000",
        "prompt_version": "v1",
        "engine_version": "1.0.0",
    }
    fields.update(overrides)
    return CacheKey(**fields)


def test_a_second_question_reuses_the_first_ones_graph() -> None:
    cache = GraphCache(max_entries=8, ttl_seconds=60, wait_seconds=30)
    builds = []

    for _ in range(2):
        cache.get_or_build(_key(), lambda: builds.append(1) or "graph")

    assert len(builds) == 1
    assert cache.hits == 1 and cache.misses == 1


def test_a_changed_schema_does_not_reuse_the_old_graph() -> None:
    """The fingerprint is what makes a changed table a different question."""
    cache = GraphCache(max_entries=8, ttl_seconds=60, wait_seconds=30)
    builds = []

    cache.get_or_build(_key(), lambda: builds.append(1) or "a")
    cache.get_or_build(
        _key(schema_fingerprint="def456"), lambda: builds.append(1) or "b"
    )

    assert len(builds) == 2


def test_every_part_of_the_key_separates_two_requests() -> None:
    """Anything that changes the graph or the generated query is in the key."""
    cache = GraphCache(max_entries=32, ttl_seconds=60, wait_seconds=30)
    changes = [
        {"connector": "SnowflakeDatabase:sales:snowflake"},
        {"database": "other"},
        {"join_fingerprint": "different"},
        {"tables": ("customers",)},
        {"schema_fingerprint": "changed"},
        {"prompt_version": "v2"},
        {"engine_version": "1.1.0"},
    ]
    builds = []

    cache.get_or_build(_key(), lambda: builds.append(1) or "base")
    for change in changes:
        cache.get_or_build(_key(**change), lambda: builds.append(1) or "other")

    assert len(builds) == 1 + len(changes)


def test_an_entry_is_dropped_once_it_is_too_old() -> None:
    """A table can be added or dropped at any time and nothing announces it."""
    cache = GraphCache(max_entries=8, ttl_seconds=0.05, wait_seconds=30)
    builds = []

    cache.get_or_build(_key(), lambda: builds.append(1) or "a")
    time.sleep(0.1)
    cache.get_or_build(_key(), lambda: builds.append(1) or "b")

    assert len(builds) == 2


def test_the_cache_stays_within_its_bound() -> None:
    """An entry holds a data snapshot, so an unbounded cache is a leak."""
    cache = GraphCache(max_entries=3, ttl_seconds=60, wait_seconds=30)

    for i in range(10):
        cache.get_or_build(_key(database=f"db{i}"), lambda: "graph")

    assert len(cache._entries) == 3


def test_the_least_recently_used_entry_is_the_one_dropped() -> None:
    cache = GraphCache(max_entries=2, ttl_seconds=60, wait_seconds=30)
    cache.get_or_build(_key(database="a"), lambda: "a")
    cache.get_or_build(_key(database="b"), lambda: "b")

    cache.get_or_build(_key(database="a"), lambda: "a")
    cache.get_or_build(_key(database="c"), lambda: "c")

    assert cache._fresh(_key(database="a")) == "a"
    assert cache._fresh(_key(database="b")) is None


def test_concurrent_callers_wait_on_one_build() -> None:
    """Without this a burst of follow-ups each reads the same tables."""
    cache = GraphCache(max_entries=8, ttl_seconds=60, wait_seconds=30)
    builds = []
    started = threading.Event()

    def slow_build() -> str:
        builds.append(1)
        started.set()
        time.sleep(0.2)
        return "graph"

    threads = [
        threading.Thread(target=lambda: cache.get_or_build(_key(), slow_build))
        for _ in range(5)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)

    assert len(builds) == 1


def test_an_invalidated_entry_is_built_again() -> None:
    cache = GraphCache(max_entries=8, ttl_seconds=60, wait_seconds=30)
    builds = []

    cache.get_or_build(_key(), lambda: builds.append(1) or "a")
    cache.invalidate(_key())
    cache.get_or_build(_key(), lambda: builds.append(1) or "b")

    assert len(builds) == 2


def test_different_join_paths_do_not_share_a_graph() -> None:
    """Join paths become the graph's edges, so they change its shape."""
    from gsf.retrieval.kumo.graph_cache import join_fingerprint

    one = [
        {
            "path": [
                {
                    "source_table": "orders",
                    "source_column": "cid",
                    "target_table": "customers",
                    "target_column": "cid",
                }
            ]
        }
    ]
    other = [
        {
            "path": [
                {
                    "source_table": "orders",
                    "source_column": "sid",
                    "target_table": "stores",
                    "target_column": "sid",
                }
            ]
        }
    ]

    assert join_fingerprint(one) != join_fingerprint(other)
    assert join_fingerprint(None) == join_fingerprint([])


def test_the_join_name_does_not_depend_on_the_order_hops_arrive() -> None:
    """The set of joins decides the graph, not the order a traversal found them."""
    from gsf.retrieval.kumo.graph_cache import join_fingerprint

    a = {
        "path": [
            {
                "source_table": "a",
                "source_column": "x",
                "target_table": "b",
                "target_column": "x",
            }
        ]
    }
    b = {
        "path": [
            {
                "source_table": "c",
                "source_column": "y",
                "target_table": "d",
                "target_column": "y",
            }
        ]
    }

    assert join_fingerprint([a, b]) == join_fingerprint([b, a])


def test_two_connections_to_the_same_database_name_are_told_apart() -> None:
    """Two deployments can describe one catalog while reading different data."""
    from gsf.retrieval.kumo.predictor import _connector_identity

    class Snowflake:
        database_name = "sales"

        def __init__(self, account: str) -> None:
            self._connect_kwargs = {"account": account}
            self._warehouse = "COMPUTE_WH"

    assert _connector_identity([Snowflake("acme")]) != _connector_identity(
        [Snowflake("other")]
    )
    assert _connector_identity([Snowflake("acme")]) == _connector_identity(
        [Snowflake("acme")]
    )


def test_a_connection_secret_does_not_appear_in_the_key() -> None:
    """A key is written to logs; a connection string carries a password."""
    from gsf.retrieval.kumo.predictor import _connector_identity

    class Postgres:
        database_name = "sales"
        _connection_string = "postgresql://user:hunter2@host/db"

    identity = _connector_identity([Postgres()])

    assert "hunter2" not in identity
    assert len(identity) == 64


def test_examples_are_not_shared_between_questions() -> None:
    """Examples are retrieved per question, so holding them answers the wrong one."""
    import dataclasses

    from gsf.retrieval.kumo.predictor import PredictionContext
    from gsf.retrieval.kumo.telemetry import GraphIdentity

    held = PredictionContext(
        kumo_model=None,
        connector=None,
        graph_ddl="",
        graph_edges=[],
        graph_col_stypes={},
        time_columns={},
        table_names={},
        entity_ids={},
        examples=[{"question": "first", "pql": "PREDICT a"}],
        column_reference="",
        identity=GraphIdentity(),
    )

    for_this_request = dataclasses.replace(
        held, examples=[{"question": "second", "pql": "PREDICT b"}]
    )

    assert for_this_request.examples[0]["question"] == "second"
    assert held.examples[0]["question"] == "first"


def test_two_duckdb_files_with_the_same_database_name_are_told_apart(
    tmp_path: Path,
) -> None:
    """DuckDB names a database after its file stem, so /prod and /staging collide."""
    import duckdb

    from gsf.connectors.duckdb import DuckDBDatabase
    from gsf.retrieval.kumo.predictor import _connector_identity

    connectors = []
    for deployment in ("prod", "staging"):
        path = tmp_path / deployment / "sales.duckdb"
        path.parent.mkdir()
        duckdb.connect(str(path)).close()
        connectors.append(DuckDBDatabase(str(path)))

    assert connectors[0].database_name == connectors[1].database_name == "sales"
    assert _connector_identity([connectors[0]]) != _connector_identity([connectors[1]])


def test_a_connector_that_cannot_say_where_it_points_is_not_cached() -> None:
    """A connector added later must not inherit another deployment's graph."""
    from gsf.retrieval.kumo.predictor import _cache_key, _connector_identity

    class Anonymous:
        database_name = "sales"

    assert _connector_identity([Anonymous()]) is None
    assert (
        _cache_key(
            [Anonymous()],
            [{"name": "orders", "database_name": "sales", "columns": []}],
            None,
        )
        is None
    )


def test_kyuubi_is_identified_by_its_parsed_settings() -> None:
    """Kyuubi keeps its host and catalog in _settings rather than named fields."""
    from gsf.retrieval.kumo.predictor import _connector_identity

    class Kyuubi:
        database_name = "sales"

        def __init__(self, host: str) -> None:
            self._settings = {"host": host, "catalog": "main"}
            self._catalog = "main"

    assert _connector_identity([Kyuubi("a.internal")]) != _connector_identity(
        [Kyuubi("b.internal")]
    )


def test_one_duckdb_file_reached_two_ways_is_one_source(tmp_path: Path) -> None:
    """A relative and an absolute path to the same file must share a cache entry."""
    import os

    import duckdb

    from gsf.connectors.duckdb import DuckDBDatabase
    from gsf.retrieval.kumo.predictor import _connector_identity

    path = tmp_path / "sales.duckdb"
    duckdb.connect(str(path)).close()

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        relative = DuckDBDatabase("sales.duckdb")
    finally:
        os.chdir(cwd)

    assert _connector_identity([relative]) == _connector_identity(
        [DuckDBDatabase(str(path))]
    )


def test_two_in_memory_duckdbs_are_not_one_source() -> None:
    """Each in-memory database is its own; neither holds the other's tables."""
    from gsf.connectors.duckdb import DuckDBDatabase
    from gsf.retrieval.kumo.predictor import _connector_identity

    first = DuckDBDatabase(":memory:", read_only=False)
    second = DuckDBDatabase(":memory:", read_only=False)

    assert first.database_name == second.database_name
    assert _connector_identity([first]) != _connector_identity([second])
    assert _connector_identity([first]) == _connector_identity([first])


def test_a_build_does_not_hold_up_a_question_about_other_tables() -> None:
    """One slow build must not become a queue every other request waits in."""
    cache = GraphCache(max_entries=8, ttl_seconds=60, wait_seconds=30)
    inside = threading.Barrier(3, timeout=5)

    def build() -> str:
        inside.wait()
        return "graph"

    threads = [
        threading.Thread(target=cache.get_or_build, args=(_key(tables=(name,)), build))
        for name in ("a", "b", "c")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)

    assert cache.misses == 3
    assert not any(thread.is_alive() for thread in threads)


def test_exactly_one_of_a_burst_is_told_it_built() -> None:
    """Telemetry reports a hit or a miss, so only one caller may claim the build."""
    cache = GraphCache(max_entries=8, ttl_seconds=60, wait_seconds=30)
    reported: list[bool] = []
    lock = threading.Lock()

    def caller() -> None:
        _, built_here = cache.get_or_build(_key(), lambda: time.sleep(0.1) or "graph")
        with lock:
            reported.append(built_here)

    threads = [threading.Thread(target=caller) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)

    assert reported.count(True) == 1
    assert reported.count(False) == 5


def test_a_failed_build_does_not_leak_its_lock() -> None:
    """Only a stored entry is ever evicted, so a lock kept on failure is kept forever."""
    cache = GraphCache(max_entries=8, ttl_seconds=60, wait_seconds=30)

    def failing() -> str:
        raise RuntimeError("the warehouse was unreachable")

    for _ in range(50):
        with pytest.raises(RuntimeError):
            cache.get_or_build(_key(), failing)

    assert cache._flights == {}


def test_a_failed_build_leaves_the_next_request_free_to_try() -> None:
    cache = GraphCache(max_entries=8, ttl_seconds=60, wait_seconds=30)
    attempts = []

    def flaky() -> str:
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("the warehouse was unreachable")
        return "graph"

    with pytest.raises(RuntimeError):
        cache.get_or_build(_key(), flaky)
    value, built_here = cache.get_or_build(_key(), flaky)

    assert (value, built_here) == ("graph", True)


def test_an_eviction_does_not_orphan_a_build_in_progress() -> None:
    """A lock dropped mid-build lets the next caller build the same graph alongside."""
    cache = GraphCache(max_entries=1, ttl_seconds=60, wait_seconds=30)
    building = threading.Event()
    may_finish = threading.Event()
    builds: list[str] = []

    def slow() -> str:
        builds.append("slow")
        building.set()
        may_finish.wait(5)
        return "slow graph"

    slow_key = _key(tables=("slow",))
    thread = threading.Thread(target=cache.get_or_build, args=(slow_key, slow))
    thread.start()
    building.wait(5)

    for name in ("a", "b"):
        cache.get_or_build(_key(tables=(name,)), lambda: "other")

    held_during_build = cache._flights.get(slow_key)
    may_finish.set()
    thread.join(5)

    assert held_during_build is not None
    assert cache._flights == {}


def test_a_burst_against_an_unreachable_warehouse_costs_one_attempt() -> None:
    """Six waiters each retrying would take six connection timeouts, in series."""
    cache = GraphCache(max_entries=8, ttl_seconds=60, wait_seconds=30)
    builds: list[int] = []
    counting = threading.Lock()

    def unreachable() -> str:
        with counting:
            builds.append(1)
        time.sleep(0.2)
        raise RuntimeError("the warehouse was unreachable")

    raised: list[BaseException] = []

    def caller() -> None:
        try:
            cache.get_or_build(_key(), unreachable)
        except BaseException as error:
            raised.append(error)

    threads = [threading.Thread(target=caller) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)

    assert len(builds) == 1
    assert len(raised) == 6
    assert all(str(error) == "the warehouse was unreachable" for error in raised)


def test_a_failure_is_not_held_against_the_next_request() -> None:
    """A warehouse unreachable a moment ago may not be now."""
    cache = GraphCache(max_entries=8, ttl_seconds=60, wait_seconds=30)
    attempts: list[int] = []

    def flaky() -> str:
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("the warehouse was unreachable")
        return "graph"

    with pytest.raises(RuntimeError):
        cache.get_or_build(_key(), flaky)

    assert cache.get_or_build(_key(), flaky) == ("graph", True)


def test_a_failed_burst_does_not_count_as_six_misses() -> None:
    """One build attempt happened, so one miss is what was paid for."""
    cache = GraphCache(max_entries=8, ttl_seconds=60, wait_seconds=30)

    def unreachable() -> str:
        time.sleep(0.2)
        raise RuntimeError("down")

    def caller() -> None:
        try:
            cache.get_or_build(_key(), unreachable)
        except RuntimeError:
            pass

    threads = [threading.Thread(target=caller) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)

    assert cache.misses == 1


def test_a_request_does_not_wait_forever_on_a_hung_build() -> None:
    """The build budget bounds a build only between tables, not within one read."""
    cache = GraphCache(max_entries=8, ttl_seconds=60, wait_seconds=0.2)
    hung = threading.Event()
    building = threading.Event()

    def never_returns() -> str:
        building.set()
        hung.wait(10)
        return "graph"

    holder = threading.Thread(target=cache.get_or_build, args=(_key(), never_returns))
    holder.start()
    building.wait(5)

    try:
        with pytest.raises(BuildTimedOut, match="already doing it"):
            cache.get_or_build(_key(), never_returns)
    finally:
        hung.set()
        holder.join(10)


def test_giving_up_does_not_read_the_same_tables_again() -> None:
    """A warehouse that cannot answer one read is not helped by a second."""
    cache = GraphCache(max_entries=8, ttl_seconds=60, wait_seconds=0.2)
    hung = threading.Event()
    building = threading.Event()
    builds: list[int] = []

    def never_returns() -> str:
        builds.append(1)
        building.set()
        hung.wait(10)
        return "graph"

    holder = threading.Thread(target=cache.get_or_build, args=(_key(), never_returns))
    holder.start()
    building.wait(5)

    try:
        for _ in range(3):
            with pytest.raises(BuildTimedOut):
                cache.get_or_build(_key(), never_returns)
        assert builds == [1]
    finally:
        hung.set()
        holder.join(10)


def test_giving_up_leaves_the_flight_for_whoever_is_still_building() -> None:
    cache = GraphCache(max_entries=8, ttl_seconds=60, wait_seconds=0.2)
    hung = threading.Event()
    building = threading.Event()

    def slow() -> str:
        building.set()
        hung.wait(10)
        return "graph"

    holder = threading.Thread(target=cache.get_or_build, args=(_key(), slow))
    holder.start()
    building.wait(5)
    with pytest.raises(BuildTimedOut):
        cache.get_or_build(_key(), slow)

    hung.set()
    holder.join(10)

    assert cache._flights == {}
    assert cache.get_or_build(_key(), lambda: "rebuilt") == ("graph", False)


def test_the_wait_is_the_budget_the_build_was_given() -> None:
    """Two requests coalescing cannot be under different limits."""
    from gsf.retrieval.kumo.budget import Budget
    from gsf.retrieval.kumo.predictor import _GRAPH_CACHE

    assert _GRAPH_CACHE._wait == Budget.from_env().max_seconds


def test_a_budget_is_the_deployments_not_the_callers() -> None:
    """If it ever became per-request it would have to join the cache key."""
    import inspect

    from gsf.retrieval.kumo import predictor

    assert (
        "budget"
        not in inspect.signature(predictor._build_context_within_budget).parameters
    )
    assert (
        "budget" not in inspect.signature(predictor.build_prediction_context).parameters
    )
    assert "Spend(Budget.from_env())" in inspect.getsource(
        predictor._build_context_within_budget
    )


def test_the_order_the_catalog_lists_columns_in_is_part_of_the_schema() -> None:
    """That order reaches the DDL the model reads, so two orders are two schemas."""
    forwards = [{"name": "t", "columns": [{"name": "a"}, {"name": "b"}]}]
    backwards = [{"name": "t", "columns": [{"name": "b"}, {"name": "a"}]}]

    assert catalog_fingerprint(forwards) != catalog_fingerprint(backwards)


def test_how_many_columns_the_catalog_lists_is_part_of_the_schema() -> None:
    once = [{"name": "t", "columns": [{"name": "a"}]}]
    twice = [{"name": "t", "columns": [{"name": "a"}, {"name": "a"}]}]

    assert catalog_fingerprint(once) != catalog_fingerprint(twice)


def test_a_column_gaining_a_description_is_a_new_schema() -> None:
    """A description changes no graph but does change the prompt built from it."""
    bare = [{"name": "t", "columns": [{"name": "a"}]}]
    documented = [{"name": "t", "columns": [{"name": "a", "description": "the id"}]}]

    assert catalog_fingerprint(bare) != catalog_fingerprint(documented)


def test_a_build_that_returns_a_failure_is_not_kept() -> None:
    """Returned failures were stored like successes and served as hits."""
    cache = GraphCache(max_entries=8, ttl_seconds=60, wait_seconds=30)
    builds = []

    def failing() -> dict[str, str]:
        builds.append(1)
        return {"error": "no tables"}

    for _ in range(3):
        value, _ = cache.get_or_build(
            _key(), failing, worth_keeping=lambda built: not isinstance(built, dict)
        )
        assert value == {"error": "no tables"}

    assert len(builds) == 3, "a failure was served from the cache"
    assert cache._entries == {}


def test_a_failure_does_not_evict_a_real_graph() -> None:
    """Counting a failure against the bound throws away something that worked."""
    cache = GraphCache(max_entries=1, ttl_seconds=60, wait_seconds=30)
    keeps = lambda built: not isinstance(built, dict)  # noqa: E731

    cache.get_or_build(_key(tables=("good",)), lambda: "graph", worth_keeping=keeps)
    cache.get_or_build(
        _key(tables=("bad",)), lambda: {"error": "x"}, worth_keeping=keeps
    )

    value, built_here = cache.get_or_build(
        _key(tables=("good",)), lambda: "rebuilt", worth_keeping=keeps
    )
    assert (value, built_here) == ("graph", False)


def test_a_successful_build_is_still_kept() -> None:
    cache = GraphCache(max_entries=8, ttl_seconds=60, wait_seconds=30)
    builds = []

    for _ in range(3):
        cache.get_or_build(
            _key(),
            lambda: builds.append(1) or "graph",
            worth_keeping=lambda built: not isinstance(built, dict),
        )

    assert len(builds) == 1


class _Weighed:
    """A cached graph that reports what holding it costs."""

    def __init__(self, nbytes: int) -> None:
        self.identity = type("Identity", (), {"bytes": nbytes})()


def test_the_cache_is_bounded_by_weight_not_only_by_count() -> None:
    """One graph holds the data it was built from, so counting entries bounds
    nothing: a few can outweigh the process."""
    cache = GraphCache(max_entries=100, ttl_seconds=60, wait_seconds=30, max_bytes=250)

    for name in ("a", "b", "c", "d"):
        cache.get_or_build(_key(tables=(name,)), lambda: _Weighed(100))

    held = sum(v.identity.bytes for _, v in cache._entries.values())
    assert held <= 250
    assert len(cache._entries) < 4


def test_the_graph_just_built_is_never_the_one_dropped() -> None:
    """Refusing to hold what was just built rebuilds it for the next question."""
    cache = GraphCache(max_entries=100, ttl_seconds=60, wait_seconds=30, max_bytes=10)

    cache.get_or_build(_key(tables=("huge",)), lambda: _Weighed(5000))

    assert len(cache._entries) == 1


def test_the_count_bound_still_applies() -> None:
    cache = GraphCache(
        max_entries=2, ttl_seconds=60, wait_seconds=30, max_bytes=1 << 40
    )

    for name in ("a", "b", "c"):
        cache.get_or_build(_key(tables=(name,)), lambda: _Weighed(1))

    assert len(cache._entries) == 2


def test_from_env_reads_both_bounds(monkeypatch: Any) -> None:
    monkeypatch.setenv("KUMO_GRAPH_CACHE_ENTRIES", "3")
    monkeypatch.setenv("KUMO_GRAPH_CACHE_BYTES", "1234")
    monkeypatch.setenv("KUMO_GRAPH_CACHE_TTL", "60")

    cache = GraphCache.from_env()

    assert (cache._max_entries, cache._max_bytes, cache._ttl) == (3, 1234, 60.0)


def test_a_nonsense_setting_falls_back_rather_than_crashing(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("KUMO_GRAPH_CACHE_ENTRIES", "not-a-number")

    assert GraphCache.from_env()._max_entries == 8
