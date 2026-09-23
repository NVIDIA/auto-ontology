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
        self.paused = False
        self.paused_during_delete: bool | None = None

    def pause(self) -> None:
        self.paused = True
        self.events.append("pause")

    def resume(self) -> None:
        self.paused = False
        self.events.append("resume")

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
        # Recorded from inside the delete: the pause has to still be in force
        # here, or a timer tick could start a pass against the half-deleted
        # layer.
        scheduler.paused_during_delete = scheduler.paused
        return None

    monkeypatch.setattr(mod, "delete_semantic_layer", _delete)


def test_delete_completes_before_the_rebuild_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = _FakeScheduler()
    _patch_delete(monkeypatch, scheduler)

    asyncio.run(mod.reset_semantic_layer_and_recompile(scheduler, "pagila"))

    assert scheduler.events == ["pause", "abort", "delete", "resume", "start"]


def test_delete_waits_for_an_in_flight_pass_to_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deleting under a running pass lets it write Terms the delete swept past."""
    scheduler = _FakeScheduler(running=True, stops_after=3)
    _patch_delete(monkeypatch, scheduler)
    monkeypatch.setattr(mod, "_ABORT_POLL_S", 0)

    asyncio.run(mod.reset_semantic_layer_and_recompile(scheduler, None))

    assert scheduler.events == ["pause", "abort", "delete", "resume", "start"]
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

    assert scheduler.events == ["pause", "abort", "delete", "resume", "start"]
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

    assert scheduler.events == ["pause", "abort", "delete", "resume"]
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

    assert scheduler.events == [
        "pause",
        "abort",
        "delete",
        "resume",
        "start",
        "trigger",
    ]


def test_the_scheduler_is_paused_across_the_whole_delete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """abort() alone leaves the loop free to start a pass mid-delete.

    ``_run_pass`` clears the abort flag before each pass, so a timer tick or
    /semantic/compile landing between the drain and the end of the delete would
    read the pre-delete catalog, compile nothing, and leave the layer deleted
    but not rebuilt.
    """
    scheduler = _FakeScheduler()
    _patch_delete(monkeypatch, scheduler)

    asyncio.run(mod.reset_semantic_layer_and_recompile(scheduler, None))

    assert scheduler.paused_during_delete is True
    assert scheduler.paused is False  # and released afterwards


def test_the_scheduler_is_resumed_even_when_the_delete_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A scheduler left paused would silently stop compiling for good."""
    scheduler = _FakeScheduler()

    def _boom(database_name: str | None) -> Any:
        raise RuntimeError("nope")

    monkeypatch.setattr(mod, "delete_semantic_layer", _boom)

    asyncio.run(mod.reset_semantic_layer_and_recompile(scheduler, None))

    assert scheduler.paused is False


def test_overlapping_resets_do_not_resume_under_each_other(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two resets at once must not have one resume while the other deletes.

    The pause is a single boolean and each reset runs as its own task, so
    without serialisation the first to finish clears the pause while the second
    is still deleting — which is exactly the window the pause closes.
    """
    scheduler = _FakeScheduler()
    deletes_in_flight = 0
    max_concurrent_deletes = 0
    paused_during_every_delete = []

    async def _delete(database_name: str | None) -> Any:
        nonlocal deletes_in_flight, max_concurrent_deletes
        deletes_in_flight += 1
        max_concurrent_deletes = max(max_concurrent_deletes, deletes_in_flight)
        paused_during_every_delete.append(scheduler.paused)
        await asyncio.sleep(0)  # yield, so an unserialised peer could interleave
        deletes_in_flight -= 1
        return None

    def _sync_delete(database_name: str | None) -> Any:
        # to_thread is patched out below, so this is only a marker.
        raise AssertionError("unused")

    monkeypatch.setattr(mod, "delete_semantic_layer", _sync_delete)
    monkeypatch.setattr(mod.asyncio, "to_thread", lambda fn, arg: _delete(arg))

    async def _both() -> None:
        await asyncio.gather(
            mod.reset_semantic_layer_and_recompile(scheduler, None),
            mod.reset_semantic_layer_and_recompile(scheduler, None),
        )

    asyncio.run(_both())

    assert max_concurrent_deletes == 1
    assert paused_during_every_delete == [True, True]
    assert scheduler.paused is False
