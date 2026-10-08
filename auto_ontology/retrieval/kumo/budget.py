# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What one prediction is allowed to spend, and what happens when it runs out.

Building a prediction graph reads whole tables into memory. Left unbounded, one
question against a large warehouse takes the process down; bounded by silently
taking the first N rows, it answers a different question than the one asked,
because an arbitrary slice chops histories and drops entities that were meant to
be scored.

So the bounds here refuse rather than truncate. A request that does not fit is
told what it exceeded and what would make it fit, which the agent can act on and
a user can understand. A request that does fit is answered on all of its data.

``KUMO_MAX_ROWS_PER_TABLE`` remains as an operator setting and is deliberately
unset: it caps rows per table before any of this, which is the truncation this
module exists to avoid, and a prediction made under it answers a narrower
question than the one asked without saying so. It is a way to keep a large
deployment inside its memory at a stated cost, not the answer to a refusal.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


class RefusedForCapacity(RuntimeError):
    """A question refused because of what answering it would cost.

    A caller that wants to treat every capacity refusal alike, to report it
    differently from a failure, has one type to catch rather than having to know
    each of the reasons a request can be too big.
    """


class BudgetExceeded(RefusedForCapacity):
    """A prediction needed more than one request is allowed to spend."""


def _readable(nbytes: int) -> str:
    """Size in the largest unit that still reads as a number, not a rounding."""
    size = float(nbytes)
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size:.0f}{unit}"
        size /= 1024
    return f"{size:.1f}GB"


def _positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    try:
        value = int(raw) if raw is not None else default
    except ValueError:
        return default
    return value if value > 0 else default


@dataclass(frozen=True)
class Budget:
    """The ceiling for one prediction, read once from the environment.

    The deployment's, not the caller's: every request is under the same limits,
    which is what lets two requests for one graph share a build. Were this ever
    to become per-request, it would have to join the cache key, or a request
    under a strict limit could be served a graph built under a loose one.
    """

    max_tables: int
    max_bytes: int
    max_seconds: float

    @classmethod
    def from_env(cls) -> "Budget":
        return cls(
            max_tables=_positive_int("KUMO_MAX_TABLES", 20),
            max_bytes=_positive_int("KUMO_MAX_BYTES", 2 * 1024**3),
            max_seconds=float(_positive_int("KUMO_MAX_BUILD_SECONDS", 300)),
        )


class Spend:
    """What one prediction has spent so far, against its budget."""

    def __init__(self, budget: Budget) -> None:
        self.budget = budget
        self.nbytes = 0
        self.tables = 0
        self._started = time.monotonic()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._started

    def check_deadline(self, doing: str) -> None:
        """Stop before starting more work than the request has time for.

        Checked between phases rather than enforced within one. A single
        warehouse query that never returns runs as long as the connector lets
        it, because cancelling it needs a per-statement timeout the connectors
        do not uniformly offer. What this bounds is everything after that: the
        remaining reads, the graph build, the link inference and the model.
        A request waiting on someone else's build is bounded separately, by the
        cache's own wait.
        """
        if self.elapsed > self.budget.max_seconds:
            raise BudgetExceeded(
                f"Preparing this prediction passed {self.budget.max_seconds:.0f}s "
                f"while {doing}. Ask about fewer tables, or narrow the question so "
                f"it needs a smaller part of the data."
            )

    def add_table(self, name: str, frame: "pd.DataFrame") -> None:
        """Record a loaded table, or refuse the request it would not fit in.

        Measured in bytes rather than rows or cells: one column of long strings
        weighs more than a hundred of integers, so a count of either bounds the
        shape of the data without bounding what it costs to hold.

        The measurement is taken after the read, so this refuses a request that
        has become too large rather than preventing a single table larger than
        the whole budget from being read at all. Bounding that needs a size the
        warehouse can be asked for before the read, which not every connector
        offers.
        """
        self.tables += 1
        if self.tables > self.budget.max_tables:
            allowed = self.budget.max_tables
            raise BudgetExceeded(
                f"This question needs more than {allowed:,} "
                f"table{'' if allowed == 1 else 's'}. Ask about a narrower part of "
                f"the data, so the tables that matter are the ones that get read."
            )
        self.nbytes += int(frame.memory_usage(deep=True).sum())
        if self.nbytes > self.budget.max_bytes:
            raise BudgetExceeded(
                f"Reading {name} took this prediction past "
                f"{_readable(self.budget.max_bytes)}, which is more than it can "
                f"hold in memory. Ask about a narrower slice of the data, so the "
                f"rows that matter are the ones that get read."
            )
