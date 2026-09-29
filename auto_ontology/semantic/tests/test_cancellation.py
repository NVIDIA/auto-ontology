# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cancelling an in-flight semantic pass stops it at the next table.

Before this, ``abort`` was only checked between databases, so a stop could take
as long as the slowest single database — 312s observed on a live store. A reset
that has to wait that long is indistinguishable from one that hung.
"""

from __future__ import annotations

from collections.abc import Iterator
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
def _clean_flag() -> Iterator[None]:
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

    def _fake_process_table(
        table: dict[str, Any], ctx: dict[str, Any], **kwargs: Any
    ) -> None:
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

    def _fake_process_table(
        table: dict[str, Any], ctx: dict[str, Any], **kwargs: Any
    ) -> None:
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


def test_cancelling_a_backlog_larger_than_the_pool_drains_cleanly() -> None:
    """A backlog far larger than the pool still finishes, with nothing compiled."""
    request_cancel()  # every table skips, before any is dispatched

    with (
        patch.object(
            pipeline, "fetch_all_tables_without_term", return_value=_tables(40)
        ),
        patch.object(
            pipeline, "fetch_table_context", return_value={"columns": [{"n": 1}]}
        ),
        patch.object(pipeline, "load_domain_summary", return_value=None),
        patch.object(pipeline, "is_distinct_value_probing_enabled", return_value=False),
        patch.object(pipeline, "reset_sampling_breaker"),
        patch.object(pipeline, "process_table", lambda *a, **k: None),
    ):
        # 40 tables against a 3-worker pool: the old blocking skip wedged here.
        count = pipeline.compile_semantic_layer("db")

    assert count == 0


def test_skipped_positions_do_not_strand_a_later_commit() -> None:
    """A skip must hand the turn on, or the next table waits on it forever."""
    queue = pipeline._OrderedCommitQueue()

    queue.skip(0)
    queue.skip(1)
    # Position 2's turn has to be reachable now that 0 and 1 gave theirs up.
    with pipeline._OrderedCommitSlot(queue, 2):
        pass

    # And the run continues in order afterwards.
    with pipeline._OrderedCommitSlot(queue, 3):
        pass


def test_a_skip_arriving_out_of_order_is_consumed_when_its_turn_comes() -> None:
    """The case waiting for a turn cannot express — it would block forever.

    Verified: with the previous, blocking implementation this test hangs.
    """
    queue = pipeline._OrderedCommitQueue()

    queue.skip(2)  # skipped before 0 and 1 have had their turns
    with pipeline._OrderedCommitSlot(queue, 0):
        pass
    with pipeline._OrderedCommitSlot(queue, 1):
        pass
    # 2 was skipped, so 3 is next without anything blocking on 2.
    with pipeline._OrderedCommitSlot(queue, 3):
        pass
