# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-table compilation emits a log line for each step it runs."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest
from pytest import LogCaptureFixture, MonkeyPatch

from auto_ontology.semantic import visit_enter
from auto_ontology.semantic.models import ColumnAttributeSpec

TABLE = {"id": "t1", "name": "orders", "schema_name": "public", "description": ""}
CTX = {"columns": [{"name": "status"}, {"name": "total"}], "fks": []}


@pytest.fixture
def _stubbed(monkeypatch: MonkeyPatch) -> None:
    """Stub every collaborator so only the logging behaviour is exercised."""
    spec = ColumnAttributeSpec(
        source_column="status",
        name="status",
        display_name="Status",
        datatype="string",
        column_description="Order status",
    )
    term = SimpleNamespace(
        name="Order", description="An order", synonyms=[], attributes=[]
    )
    assignment = SimpleNamespace(source_column="status")

    monkeypatch.setattr(visit_enter, "_resolve_connector", lambda _db: object())
    monkeypatch.setattr(
        visit_enter, "calculate_columns_profiling", lambda *_a, **_k: {}
    )
    monkeypatch.setattr(
        visit_enter,
        "suggest_potential_foreign_keys",
        lambda *_a, **_k: SimpleNamespace(suggestions=[], pk_column_names=[]),
    )
    monkeypatch.setattr(
        visit_enter, "fill_missing_column_descriptions", lambda *_a, **_k: []
    )
    monkeypatch.setattr(visit_enter, "column_attribute_specs", lambda *_a, **_k: [spec])
    monkeypatch.setattr(
        visit_enter, "extract_term", lambda *_a, **_k: SimpleNamespace(terms=[term])
    )
    monkeypatch.setattr(visit_enter, "apply_display_names_to_specs", lambda *_a: None)
    monkeypatch.setattr(
        visit_enter, "terms_with_assignments", lambda *_a: [(term, [assignment])]
    )
    monkeypatch.setattr(
        visit_enter,
        "upsert_table_term",
        lambda *_a, **_k: ("term1", "Order"),
    )
    monkeypatch.setattr(visit_enter, "merge_column_attribute", lambda **_k: None)
    monkeypatch.setattr(visit_enter, "clear_column_values", lambda *_a: None)
    monkeypatch.setattr(
        visit_enter,
        "fetch_terms_and_attributes_for_table",
        lambda _id: ([{"id": "term1", "name": "Order"}], []),
    )


def _run(embedder: Any = None) -> None:
    visit_enter.process_table(
        TABLE, CTX, domain_summary=None, embedder=embedder, database_name="mydb"
    )


def test_each_step_is_logged(_stubbed: None, caplog: LogCaptureFixture) -> None:
    embedder = SimpleNamespace(embed_term=lambda *_a: None)

    with caplog.at_level(logging.INFO, logger="auto_ontology.semantic.visit_enter"):
        _run(embedder)

    messages = [record.getMessage() for record in caplog.records]

    for expected in (
        "[orders] Sampling column values (2 columns)…",
        "[orders] Detecting foreign keys…",
        "[orders] Describing 2 undocumented column(s)…",
        "[orders] Generating terms (1 columns)…",
        "[orders] Writing 1 term(s) to the graph…",
        "[orders] Embedding 1 term(s)…",
    ):
        assert expected in messages, f"missing step log: {expected}"

    # Every step also reports how long it took.
    assert (
        sum("Sampling column values (2 columns) done in " in m for m in messages) == 1
    )


def test_steps_are_logged_in_pipeline_order(
    _stubbed: None, caplog: LogCaptureFixture
) -> None:
    embedder = SimpleNamespace(embed_term=lambda *_a: None)

    with caplog.at_level(logging.INFO, logger="auto_ontology.semantic.visit_enter"):
        _run(embedder)

    starts = [m for m in (r.getMessage() for r in caplog.records) if m.endswith("…")]
    order = [
        "Sampling column values (2 columns)",
        "Detecting foreign keys",
        "Describing 2 undocumented column(s)",
        "Generating terms (1 columns)",
        "Writing 1 term(s) to the graph",
        "Embedding 1 term(s)",
    ]
    assert [s.split("] ", 1)[1].rstrip("…") for s in starts] == order


def test_unreadable_pii_tags_skip_the_inline_embed(
    _stubbed: None, monkeypatch: MonkeyPatch, caplog: LogCaptureFixture
) -> None:
    """The embed reads catalog sample_values, so unknown PII tags skip it."""

    def _unreadable(_ids: list[str]) -> set[str]:
        raise RuntimeError("store down")

    monkeypatch.setattr(visit_enter, "pii_tagged_column_ids", _unreadable)
    embedded: list[Any] = []
    embedder = SimpleNamespace(embed_term=lambda *a: embedded.append(a))
    ctx = {"columns": [{"id": "c1", "name": "status"}], "fks": []}

    with caplog.at_level(logging.INFO, logger="auto_ontology.semantic.visit_enter"):
        visit_enter.process_table(
            TABLE, ctx, domain_summary=None, embedder=embedder, database_name="mydb"
        )

    assert embedded == []
    messages = [record.getMessage() for record in caplog.records]
    assert "[orders] Skipping inline embed — could not read PII tags" in messages


def test_sampling_skip_is_logged_without_connector(
    _stubbed: None, monkeypatch: MonkeyPatch, caplog: LogCaptureFixture
) -> None:
    """A missing connector says so rather than silently skipping profiling."""
    monkeypatch.setattr(visit_enter, "_resolve_connector", lambda _db: None)

    with caplog.at_level(logging.INFO, logger="auto_ontology.semantic.visit_enter"):
        _run()

    messages = [record.getMessage() for record in caplog.records]
    assert "[orders] Skipping value sampling — no live connector for 'mydb'" in messages
    assert not any("Sampling column values" in m for m in messages)


def test_sql_attribute_extraction_is_off_by_default(
    _stubbed: None, monkeypatch: MonkeyPatch
) -> None:
    calls: list[Any] = []
    monkeypatch.setattr(
        visit_enter, "_extract_sql_attrs_for_terms", lambda *a, **k: calls.append(a)
    )

    result = visit_enter.process_table(
        TABLE, CTX, domain_summary=None, database_name="mydb"
    )

    assert calls == []
    assert result.sql_attr_names == []


def test_sql_attribute_extraction_runs_last_when_enabled(
    _stubbed: None, monkeypatch: MonkeyPatch, caplog: LogCaptureFixture
) -> None:
    monkeypatch.setattr(
        visit_enter, "_extract_sql_attrs_for_terms", lambda *_a, **_k: ["Margin"]
    )
    embedder = SimpleNamespace(embed_term=lambda *_a: None)

    with caplog.at_level(logging.INFO, logger="auto_ontology.semantic.visit_enter"):
        result = visit_enter.process_table(
            TABLE,
            CTX,
            domain_summary=None,
            embedder=embedder,
            database_name="mydb",
            extract_table_sql_attributes=True,
        )

    starts = [m for m in (r.getMessage() for r in caplog.records) if m.endswith("…")]
    assert starts[-1] == "[orders] Extracting SQL attributes…"
    assert result.sql_attr_names == ["Margin"]


def test_sql_attribute_extraction_leaves_out_pii_columns(
    monkeypatch: MonkeyPatch,
) -> None:
    seen: list[list[str]] = []

    def _extract(_table: Any, columns: list[dict], *_a: Any) -> list[str]:
        seen.append([c["name"] for c in columns])
        return []

    monkeypatch.setattr(visit_enter, "_extract_sql_attributes_for_table", _extract)
    ctx = {
        "columns": [{"name": "price"}, {"name": "cost"}, {"name": "email"}],
        "fks": [],
    }
    attrs = {
        "Order": [
            {"source_column": "price"},
            {"source_column": "cost"},
            {"source_column": "email"},
        ],
        "Customer": [{"source_column": "email"}, {"source_column": "cost"}],
    }

    visit_enter._extract_sql_attrs_for_terms(
        TABLE,
        ctx,
        [{"id": "t1", "name": "Order"}, {"id": "t2", "name": "Customer"}],
        attrs,
        "mydb",
        pii_columns={"email"},
        profiling={"price": {"sample_values": [10, 20]}},
    )

    # Customer is left with one column once email is dropped, so it is skipped.
    assert seen == [["price", "cost"]]
