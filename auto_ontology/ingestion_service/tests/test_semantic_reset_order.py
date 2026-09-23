# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ordering of the semantic reset: stop, drain, delete, only then rebuild.

These used to overlap, and both orderings lost data. A compilation pass reads
its work list once at the start, so a pass that begins before the delete lands
sees every table still carrying a Term, compiles nothing, and leaves the
database wiped but not rebuilt.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from auto_ontology.ingestion_service import ingest as mod


class _FakeScheduler:
    """Records the order of calls, and can pretend a pass is still running."""

    def __init__(self, *, running: bool = False, stops_after: int = 0) -> None:
        self._running = running
        self._polls_until_idle = stops_after
        self.events: list[str] = []
        self.started = False

    @property
    def running(self) -> bool:
        if self._running and self._polls_until_idle <= 0:
            self._running = False
        self._polls_until_idle -= 1
        return self._running

    def abort(self) -> None:
        self.events.append("abort")

    def start(self) -> bool:
        self.events.append("start")
        self.started = True
        return True

    def trigger(self) -> None:
        self.events.append("trigger")


def _patch_delete(monkeypatch: pytest.MonkeyPatch, scheduler: _FakeScheduler) -> None:
    def _delete(database_name: str | None) -> Any:
        scheduler.events.append("delete")
        return None

    monkeypatch.setattr(mod, "delete_semantic_layer", _delete)


def test_delete_completes_before_the_rebuild_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = _FakeScheduler()
    _patch_delete(monkeypatch, scheduler)

    asyncio.run(mod.reset_semantic_layer_and_recompile(scheduler, "pagila"))

    assert scheduler.events == ["abort", "delete", "start"]


def test_delete_waits_for_an_in_flight_pass_to_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deleting under a running pass lets it write Terms the delete swept past."""
    scheduler = _FakeScheduler(running=True, stops_after=3)
    _patch_delete(monkeypatch, scheduler)
    monkeypatch.setattr(mod, "_ABORT_POLL_S", 0)

    asyncio.run(mod.reset_semantic_layer_and_recompile(scheduler, None))

    assert scheduler.events == ["abort", "delete", "start"]
    # The drain actually waited rather than falling through on the first poll.
    assert scheduler._polls_until_idle < 0


def test_a_pass_that_never_stops_does_not_block_the_reset_forever(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    scheduler = _FakeScheduler(running=True, stops_after=10**6)
    _patch_delete(monkeypatch, scheduler)
    monkeypatch.setattr(mod, "_ABORT_POLL_S", 0)
    monkeypatch.setattr(mod, "_ABORT_DRAIN_TIMEOUT_S", 0)

    asyncio.run(mod.reset_semantic_layer_and_recompile(scheduler, None))

    assert scheduler.events == ["abort", "delete", "start"]
    assert "still running" in caplog.text


def test_a_failed_delete_does_not_start_a_rebuild(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compiling against a half-deleted layer yields neither the old nor a clean one."""
    scheduler = _FakeScheduler()

    def _boom(database_name: str | None) -> Any:
        scheduler.events.append("delete")
        raise RuntimeError("nope")

    monkeypatch.setattr(mod, "delete_semantic_layer", _boom)

    asyncio.run(mod.reset_semantic_layer_and_recompile(scheduler, None))

    assert scheduler.events == ["abort", "delete"]
    assert scheduler.started is False


def test_an_already_running_scheduler_is_triggered_rather_than_started(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = _FakeScheduler()
    monkeypatch.setattr(
        scheduler, "start", lambda: scheduler.events.append("start") or False
    )
    _patch_delete(monkeypatch, scheduler)

    asyncio.run(mod.reset_semantic_layer_and_recompile(scheduler, None))

    assert scheduler.events == ["abort", "delete", "start", "trigger"]
