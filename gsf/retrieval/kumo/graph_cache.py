# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusing the graph a question already built, for the questions that follow it.

Every prediction reads its tables, builds a graph and creates a model. Asked a
follow-up about the same tables, it does all of that again, so the second
question pays the same warehouse and memory cost as the first for a graph that
has not changed.

What is safe to reuse is the built graph and the model over it. What is not is
the answer: a prediction is anchored to the moment it was asked, so a cached
result is an answer to a question about a different day.

The key is everything that can change what the graph or the generated query would
be. Two requests share an entry only when the database, the tables, the schema
the catalog reported, the prompt, the engine and the model behind the LLM all
match. A single-flight gate means the first request through builds while the
others wait for it, rather than each building its own copy of the same thing.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)


class BuildTimedOut(TimeoutError):
    """A request gave up waiting for someone else's build of the same graph."""


def _term(value: object) -> str:
    """One field, written as its length then its text.

    A fingerprint made by joining fields on a separator can be forged by a value
    that contains the separator: ["a,b", "c"] and ["a", "b,c"] join to the same
    string. Writing the length first leaves nothing for a value to imitate.
    """
    text = "" if value is None else str(value)
    return f"{len(text)}:{text}"


@dataclass
class _Flight:
    """One build, and whoever is waiting on it to finish.

    A build that returns is shared through the stored entry. A build that RAISES
    stores nothing, so without somewhere to put the failure each waiter would
    wake, find nothing, and start the same doomed build again: six concurrent
    questions against an unreachable warehouse became six builds run one after
    another, each paying the full connection timeout. Holding the failure here
    lets the requests that were waiting on that build share its outcome.

    The failure lives no longer than the flight. Once the last participant
    leaves, the next request starts a new one, because a warehouse that was
    unreachable a moment ago may not be now.
    """

    lock: threading.Lock
    waiting: int = 0
    finished: bool = False
    error: BaseException | None = None


@dataclass(frozen=True)
class CacheKey:
    """Everything that decides whether two requests may share a graph."""

    connector: str
    database: str
    tables: tuple[str, ...]
    schema_fingerprint: str
    join_fingerprint: str
    prompt_version: str
    engine_version: str


class GraphCache:
    """A bounded, expiring store of built graphs, one build per key.

    Bounded because a deployment may hold thousands of datasets and an entry
    holds its data snapshot; expiring because a table can be added or dropped at
    any time and nothing tells us when.
    """

    def __init__(
        self, max_entries: int, ttl_seconds: float, wait_seconds: float
    ) -> None:
        self._max_entries = max_entries
        self._ttl = ttl_seconds
        self._wait = wait_seconds
        self._entries: OrderedDict[CacheKey, tuple[float, Any]] = OrderedDict()
        self._flights: dict[CacheKey, _Flight] = {}
        self._guard = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _fresh(self, key: CacheKey) -> Any | None:
        stored = self._entries.get(key)
        if stored is None:
            return None
        stored_at, value = stored
        if time.monotonic() - stored_at > self._ttl:
            del self._entries[key]
            return None
        self._entries.move_to_end(key)
        return value

    def get_or_build(self, key: CacheKey, build: Callable[[], Any]) -> tuple[Any, bool]:
        """The graph for *key* and whether this call is what built it.

        Concurrent callers for one key wait on the first rather than each
        building a copy, which is what keeps a burst of follow-ups from reading
        the same tables several times over. Only one of them is told it built.
        """
        with self._guard:
            value = self._fresh(key)
            if value is not None:
                self.hits += 1
                return value, False
            flight = self._flights.get(key)
            joined = flight is not None
            if flight is None:
                flight = self._flights[key] = _Flight(threading.Lock())
            flight.waiting += 1

        try:
            if not flight.lock.acquire(timeout=self._wait):
                raise BuildTimedOut(
                    f"Preparing this prediction is taking longer than "
                    f"{self._wait:.0f}s, and another request is already doing "
                    f"it. Try again in a moment."
                )
            try:
                with self._guard:
                    value = self._fresh(key)
                    if value is not None:
                        self.hits += 1
                        return value, False
                    failure = flight.error if joined and flight.finished else None
                    if failure is None:
                        self.misses += 1
                if failure is not None:
                    raise failure
                try:
                    built = build()
                except BaseException as error:
                    with self._guard:
                        flight.finished = True
                        flight.error = error
                    raise
                with self._guard:
                    flight.finished = True
                    self._entries[key] = (time.monotonic(), built)
                    self._entries.move_to_end(key)
                    while len(self._entries) > self._max_entries:
                        self._entries.popitem(last=False)
                return built, True
            finally:
                flight.lock.release()
        finally:
            with self._guard:
                flight.waiting -= 1
                self._retire(key, flight)

    def _retire(self, key: CacheKey, flight: "_Flight") -> None:
        """Forget a flight once nobody is waiting on it.

        A flight dropped while its build is still running would let the next
        caller start a second one alongside it, and a flight kept for a build
        that failed would never be dropped at all, since only a stored entry is
        ever evicted. Counting who is waiting answers both.
        """
        if flight.waiting <= 0 and self._flights.get(key) is flight:
            del self._flights[key]

    def invalidate(self, key: CacheKey) -> None:
        """Drop an entry, so the next request for it builds again."""
        with self._guard:
            self._entries.pop(key, None)

    def clear(self) -> None:
        with self._guard:
            self._entries.clear()


def catalog_fingerprint(relevant_tables: list[dict[str, Any]]) -> str:
    """Name the schema the catalog reported, before any data is read.

    Taken from the catalog rather than from the built graph because the key has
    to be known before deciding whether to build. Every field is written as its
    length then its text, so a name containing whatever character separated them
    cannot split into two and let two different schemas share a name.

    Covers what the graph is built from: the tables, their columns, and the types
    and keys the catalog recorded for them. A description changing does not alter
    the graph but does alter the prompt, so it is included too.
    """
    parts: list[str] = []
    for table in sorted(relevant_tables or [], key=lambda t: str(t.get("name") or "")):
        if not isinstance(table, dict):
            continue
        parts.extend(
            [
                _term(table.get("name")),
                _term(table.get("schema_name")),
                _term(table.get("database_name")),
                _term(table.get("description")),
            ]
        )
        key = table.get("primary_key") or table.get("pk") or []
        key = [key] if isinstance(key, str) else list(key)
        parts.append(_term(len(key)))
        parts.extend(_term(column) for column in key)
        columns = [c for c in table.get("columns") or [] if isinstance(c, dict)]
        parts.append(_term(len(columns)))
        for column in columns:
            parts.extend(
                [
                    _term(column.get("name")),
                    _term(column.get("data_type")),
                    _term(column.get("description")),
                ]
            )
    return hashlib.sha256("".join(parts).encode("utf-8")).hexdigest()


def join_fingerprint(join_paths: list[dict[str, Any]] | None) -> str:
    """Name the joins a request was given, since they become the graph's edges.

    Two requests over the same tables with different join paths build differently
    shaped graphs, so a key blind to them would answer one from the other's.
    """
    hops: list[str] = []
    for entry in join_paths or []:
        if not isinstance(entry, dict):
            continue
        for hop in entry.get("path") or []:
            if not isinstance(hop, dict):
                continue
            hops.append(
                "".join(
                    _term(hop.get(field))
                    for field in (
                        "source_table",
                        "source_column",
                        "target_table",
                        "target_column",
                    )
                )
            )
    return hashlib.sha256("".join(sorted(hops)).encode("utf-8")).hexdigest()


def built_graph_fingerprint(graph: Any) -> str:
    """Name the graph that was actually built, after inference has run.

    The catalog fingerprint names what a request was given; this names what came
    of it. They differ whenever the engine's own inference decides something the
    catalog did not say: a column's semantic type, a time column, or a link. Two
    runs over one schema can build different graphs that way, and when a
    prediction is wrong it is this graph, not the catalog, that produced it.

    Columns are read in the order the graph holds them, since that order reaches
    the model. Edges are sorted, because they are a set.
    """
    parts: list[str] = []
    for name in sorted(getattr(graph, "tables", {})):
        table = graph[name]
        columns = list(getattr(table, "columns", []))
        parts.extend([_term(name), _term(len(columns))])
        for column in columns:
            parts.extend(
                [
                    _term(getattr(column, "name", "")),
                    _term(getattr(column, "stype", "")),
                    _term(getattr(column, "dtype", "")),
                ]
            )
        declared = tuple(getattr(table, "primary_key_columns", ()) or ())
        parts.append(_term(len(declared)))
        parts.extend(_term(column) for column in declared)
        for attribute in ("primary_key", "time_column", "end_time_column"):
            held = getattr(table, attribute, None)
            parts.append(_term(getattr(held, "name", held)))

    edges = sorted(
        (
            str(getattr(edge, "src_table", "")),
            str(getattr(edge, "fkey", "")),
            str(getattr(edge, "dst_table", "")),
        )
        for edge in getattr(graph, "edges", [])
    )
    parts.append(_term(len(edges)))
    for src, fkey, dst in edges:
        parts.extend([_term(src), _term(fkey), _term(dst)])

    return hashlib.sha256("".join(parts).encode("utf-8")).hexdigest()
