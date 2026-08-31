# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a prediction says about itself, and what it must not say."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

from pytest import LogCaptureFixture

from gsf.retrieval.kumo.graph_cache import built_graph_fingerprint
from gsf.retrieval.kumo.telemetry import (
    CACHE_HIT,
    CACHE_MISS,
    GraphIdentity,
    RunRecord,
    emit,
    llm_model_name,
    redact_literals,
)


def _column(name: str, stype: str = "numerical", dtype: str = "int64") -> object:
    return SimpleNamespace(name=name, stype=stype, dtype=dtype)


def _table(
    columns: list[object],
    primary_key: str | None = None,
    time_column: str | None = None,
) -> object:
    return SimpleNamespace(
        columns=columns,
        primary_key=SimpleNamespace(name=primary_key) if primary_key else None,
        primary_key_columns=(),
        time_column=SimpleNamespace(name=time_column) if time_column else None,
        end_time_column=None,
    )


class _Graph:
    def __init__(self, tables: dict[str, object], edges: list[object]) -> None:
        self.tables = tables
        self.edges = edges

    def __getitem__(self, name: str) -> object:
        return self.tables[name]


def _edge(src: str, fkey: str, dst: str) -> object:
    return SimpleNamespace(src_table=src, fkey=fkey, dst_table=dst)


def _graph(**overrides: object) -> _Graph:
    tables = {
        "customers": _table([_column("cid"), _column("seg", "categorical")], "cid"),
        "orders": _table(
            [_column("oid"), _column("cid"), _column("ts", "timestamp")],
            "oid",
            "ts",
        ),
    }
    tables.update(overrides.get("tables", {}))  # type: ignore[arg-type]
    return _Graph(
        tables, list(overrides.get("edges", [_edge("orders", "cid", "customers")]))
    )  # type: ignore[arg-type]


def test_the_same_graph_fingerprints_the_same_way() -> None:
    assert built_graph_fingerprint(_graph()) == built_graph_fingerprint(_graph())


def test_a_column_whose_inferred_type_changed_is_a_different_graph() -> None:
    """Inference decides stypes, and a different stype predicts differently."""
    changed = _table([_column("cid"), _column("seg", "text")], "cid")

    assert built_graph_fingerprint(_graph()) != built_graph_fingerprint(
        _graph(tables={"customers": changed})
    )


def test_a_lost_time_column_is_a_different_graph() -> None:
    """Without its time column a table can anchor no forecast."""
    undated = _table(
        [_column("oid"), _column("cid"), _column("ts", "timestamp")], "oid"
    )

    assert built_graph_fingerprint(_graph()) != built_graph_fingerprint(
        _graph(tables={"orders": undated})
    )


def test_a_missing_link_is_a_different_graph() -> None:
    assert built_graph_fingerprint(_graph()) != built_graph_fingerprint(
        _graph(edges=[])
    )


def test_edges_are_a_set_not_a_sequence() -> None:
    """The order a traversal reported links in does not change the graph."""
    both = [_edge("orders", "cid", "customers"), _edge("orders", "sid", "sellers")]

    assert built_graph_fingerprint(_graph(edges=both)) == built_graph_fingerprint(
        _graph(edges=list(reversed(both)))
    )


def test_a_table_name_containing_the_separator_cannot_forge_another_graph() -> None:
    """Length-prefixed terms leave nothing for a value to imitate."""
    one = _graph(tables={"a:b": _table([_column("x")], "x")})
    other = _graph(tables={"a": _table([_column("b:x")], "b:x")})

    assert built_graph_fingerprint(one) != built_graph_fingerprint(other)


def test_a_run_is_written_as_one_parseable_line(caplog: LogCaptureFixture) -> None:
    record = RunRecord(
        outcome="answered",
        llm_model="gpt-4.1-mini",
        pql="PREDICT COUNT(orders.*, 0, 30) FOR customers.cid IN (1)",
        attempts=2,
        entities=1,
        rows_returned=1,
        seconds=4.2,
        graph=GraphIdentity(
            fingerprint="abc", engine_version="2.29.0", cache=CACHE_HIT
        ),
    )

    with caplog.at_level(logging.INFO, logger="gsf.retrieval.kumo.telemetry"):
        emit(record)

    written = json.loads(caplog.records[0].getMessage().removeprefix("kumo.run "))

    assert written["outcome"] == "answered"
    assert written["attempts"] == 2
    assert written["graph"]["fingerprint"] == "abc"
    assert written["graph"]["cache"] == CACHE_HIT


def test_a_record_that_cannot_be_written_does_not_fail_the_request(
    caplog: LogCaptureFixture,
) -> None:
    """Telemetry describes a prediction; it must never be what breaks one."""

    class Unserializable:
        def __repr__(self) -> str:
            raise RuntimeError("no")

    record = RunRecord(outcome="answered")
    record.entities = Unserializable()  # type: ignore[assignment]

    with caplog.at_level(logging.INFO, logger="gsf.retrieval.kumo.telemetry"):
        emit(record)

    assert not any(r.getMessage().startswith("kumo.run ") for r in caplog.records)


def test_a_reused_graph_still_names_the_build_it_came_from() -> None:
    built = GraphIdentity(fingerprint="abc", build_seconds=9.0, cache=CACHE_MISS)

    reused = built.reused()

    assert reused.cache == CACHE_HIT
    assert reused.fingerprint == "abc"
    assert reused.build_seconds == 9.0


def test_the_model_is_named_under_whichever_field_its_client_uses() -> None:
    assert llm_model_name(SimpleNamespace(model_name="a")) == "a"
    assert llm_model_name(SimpleNamespace(model="b")) == "b"
    assert llm_model_name(SimpleNamespace(deployment_name="c")) == "c"
    assert llm_model_name(SimpleNamespace()) == ""


def test_a_count_is_fingerprinted_without_raising() -> None:
    """_term takes any value, not only text: counts go through str() first."""
    assert built_graph_fingerprint(_graph())


def test_the_entities_a_query_named_are_not_written_to_the_log() -> None:
    """An entity list is a list of real customers."""
    redacted = redact_literals(
        "PREDICT COUNT(orders.*, 0, 30) FOR customers.email IN "
        "('alice@acme.com', 'bob@acme.com', 'carol@acme.com')"
    )

    assert "acme.com" not in redacted
    assert "IN (3 values)" in redacted
    assert "COUNT(orders.*, 0, 30)" in redacted
    assert "customers.email" in redacted


def test_a_quoted_filter_is_not_written_to_the_log() -> None:
    """A WHERE literal is whatever the question named."""
    redacted = redact_literals(
        "PREDICT SUM(orders.amt, 0, 30) FOR EACH customers.cid "
        "WHERE customers.name = 'Jane Roe'"
    )

    assert "Jane Roe" not in redacted
    assert "customers.name = '?'" in redacted


def test_an_apostrophe_inside_a_value_does_not_end_it() -> None:
    """A doubled quote is an escaped quote, not the close of the literal."""
    redacted = redact_literals(
        "PREDICT COUNT(orders.*, 0, 30) FOR customers.name IN ('O''Brien', 'Ng')"
    )

    assert "Brien" not in redacted
    assert "IN (2 values)" in redacted


def test_a_comma_inside_a_value_is_not_counted_as_a_separator() -> None:
    redacted = redact_literals(
        "PREDICT COUNT(orders.*, 0, 30) FOR customers.city IN ('Paris, TX', 'Rome')"
    )

    assert "IN (2 values)" in redacted


def test_the_windows_that_give_a_query_its_shape_are_kept() -> None:
    """A time window names nobody, and it is what a misread question shows up in."""
    kept = redact_literals(
        "PREDICT SUM(orders.amt, 0, 90) FOR EACH customers.cid WHERE orders.amt > 5000"
    )

    assert kept == (
        "PREDICT SUM(orders.amt, 0, 90) FOR EACH customers.cid WHERE orders.amt > 5000"
    )


def test_an_empty_entity_list_reads_as_empty() -> None:
    assert "IN (0 values)" in redact_literals("PREDICT x FOR customers.cid IN ()")


def test_nothing_to_redact_leaves_the_query_alone() -> None:
    query = "PREDICT COUNT(orders.*, 0, 30) FOR EACH customers.cid"

    assert redact_literals(query) == query
    assert redact_literals("") == ""


def test_the_recorded_query_is_the_redacted_one() -> None:
    """The redaction has to be where the record is made, not left to the caller."""
    import inspect

    from gsf.retrieval.kumo import predictor

    source = inspect.getsource(predictor._run_prediction)

    assert "record.pql = redact_literals(" in source


def test_a_bracket_inside_a_value_does_not_hide_it() -> None:
    """Strings go first, so a paren in a value is gone before lists are counted."""
    redacted = redact_literals(
        "PREDICT COUNT(orders.*, 0, 30) FOR customers.org IN ('Acme (US)', 'Beta')"
    )

    assert "Acme" not in redacted
    assert "IN (2 values)" in redacted


def test_the_order_columns_are_held_in_is_part_of_the_graph() -> None:
    """That order reaches the model, so two orders are two graphs."""
    forwards = _table([_column("a"), _column("b", "categorical")], "a")
    backwards = _table([_column("b", "categorical"), _column("a")], "a")

    assert built_graph_fingerprint(
        _graph(tables={"customers": forwards})
    ) != built_graph_fingerprint(_graph(tables={"customers": backwards}))


def test_how_many_columns_a_table_has_is_part_of_the_graph() -> None:
    one = _table([_column("a")], "a")
    two = _table([_column("a"), _column("a")], "a")

    assert built_graph_fingerprint(
        _graph(tables={"customers": one})
    ) != built_graph_fingerprint(_graph(tables={"customers": two}))


def test_how_many_edges_a_graph_has_is_part_of_it() -> None:
    """A count that stops counting lets one link stand in for three."""
    one = [_edge("orders", "cid", "customers")]
    three = one + [
        _edge("orders", "sid", "sellers"),
        _edge("orders", "pid", "products"),
    ]

    assert built_graph_fingerprint(_graph(edges=one)) != built_graph_fingerprint(
        _graph(edges=three)
    )


def test_a_repeated_edge_is_not_the_same_as_a_single_one() -> None:
    single = [_edge("orders", "cid", "customers")]

    assert built_graph_fingerprint(_graph(edges=single)) != built_graph_fingerprint(
        _graph(edges=single * 2)
    )


def test_an_empty_column_name_cannot_stand_in_for_a_missing_one() -> None:
    """Empty terms are where a naive encoding most often collides."""
    named = _table([_column("a"), _column("")], "a")
    fewer = _table([_column("a")], "a")

    assert built_graph_fingerprint(
        _graph(tables={"customers": named})
    ) != built_graph_fingerprint(_graph(tables={"customers": fewer}))
