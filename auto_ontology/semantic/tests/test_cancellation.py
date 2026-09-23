# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cancelling an in-flight semantic pass stops it at the next table.

Before this, ``abort`` was only checked between databases, so a stop could take
as long as the slowest single database — 312s observed on a live store. A reset
that has to wait that long is indistinguishable from one that hung.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from auto_ontology.semantic import pipeline
from auto_ontology.semantic.cancellation import (
    clear_cancel,
    is_cancelled,
    request_cancel,
)


@pytest.fixture(autouse=True)
def _clean_flag():
    clear_cancel()
    yield
    clear_cancel()


def test_flag_round_trips() -> None:
    assert is_cancelled() is False
    request_cancel()
    assert is_cancelled() is True
    clear_cancel()
    assert is_cancelled() is False


def _tables(n: int) -> list[dict[str, Any]]:
    return [{"id": f"t{i}", "name": f"table_{i}"} for i in range(n)]


def test_queued_tables_are_skipped_once_cancelled() -> None:
    """The backlog becomes no-ops, so the pool drains in one table's time."""
    processed: list[str] = []

    def _fake_process_table(table, ctx, **kwargs):
        processed.append(table["name"])
        # Cancel partway through, as a reset would.
        if len(processed) == 1:
            request_cancel()
        return None

    with (
        patch.object(
            pipeline, "fetch_all_tables_without_term", return_value=_tables(25)
        ),
        patch.object(
            pipeline, "fetch_table_context", return_value={"columns": [{"n": 1}]}
        ),
        patch.object(pipeline, "load_domain_summary", return_value=None),
        patch.object(pipeline, "is_distinct_value_probing_enabled", return_value=False),
        patch.object(pipeline, "reset_sampling_breaker"),
        patch.object(pipeline, "process_table", _fake_process_table),
    ):
        count = pipeline.compile_semantic_layer("db")

    # Far fewer than the 25 queued: the ones already in flight finish, the rest
    # are dropped. The exact number depends on pool width, so assert the shape.
    assert len(processed) < 25
    assert count == 0


def test_an_uncancelled_pass_still_processes_everything() -> None:
    processed: list[str] = []

    def _fake_process_table(table, ctx, **kwargs):
        processed.append(table["name"])
        return None

    with (
        patch.object(
            pipeline, "fetch_all_tables_without_term", return_value=_tables(8)
        ),
        patch.object(
            pipeline, "fetch_table_context", return_value={"columns": [{"n": 1}]}
        ),
        patch.object(pipeline, "load_domain_summary", return_value=None),
        patch.object(pipeline, "is_distinct_value_probing_enabled", return_value=False),
        patch.object(pipeline, "reset_sampling_breaker"),
        patch.object(pipeline, "process_table", _fake_process_table),
    ):
        pipeline.compile_semantic_layer("db")

    assert len(processed) == 8


def test_scheduler_abort_requests_cancellation() -> None:
    """``abort`` has to reach the worker threads, not just the loop's own flag."""
    from auto_ontology.ingestion_service.semantic_scheduler import SemanticScheduler

    scheduler = SemanticScheduler()
    assert is_cancelled() is False

    scheduler.abort()

    assert is_cancelled() is True
    assert scheduler.aborting is True
