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
        description="Order status",
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
        lambda *_a, **_k: SimpleNamespace(suggestions=[]),
    )
    monkeypatch.setattr(visit_enter, "column_attribute_specs", lambda *_a, **_k: [spec])
    monkeypatch.setattr(
        visit_enter, "extract_term", lambda *_a, **_k: SimpleNamespace(terms=[term])
    )
    monkeypatch.setattr(visit_enter, "apply_display_names_to_specs", lambda *_a: None)
    monkeypatch.setattr(
        visit_enter, "_terms_with_assignments", lambda *_a: [(term, [assignment])]
    )
    monkeypatch.setattr(
        visit_enter,
        "upsert_table_term",
        lambda *_a, **_k: ("term1", "Order"),
    )
    monkeypatch.setattr(visit_enter, "merge_column_attribute", lambda **_k: None)
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
        "[orders] Generating terms and descriptions (1 columns)…",
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
        "Generating terms and descriptions (1 columns)",
        "Writing 1 term(s) to the graph",
        "Embedding 1 term(s)",
    ]
    assert [s.split("] ", 1)[1].rstrip("…") for s in starts] == order


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
