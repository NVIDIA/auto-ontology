# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment flags for multi-candidate SQL generation.

Every default reproduces single-candidate behaviour, so an unset environment
leaves the graph exactly as it was before candidates existed. Values are read
per call rather than captured at import, so a sweep can change one between
questions without reloading the module.
"""

import logging
import os

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer; using %d", name, raw, default)
        return default


def _float(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; using %s", name, raw, default)
        return default


def _bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in _TRUTHY


def num_candidates() -> int:
    """How many SQL candidates to generate per question (``BIRD_NCAND``).

    1 keeps the single-call path, retries included.
    """
    return max(1, _int("BIRD_NCAND", 1))


def candidate_temperature() -> float:
    """Sampling temperature for slots 1..N-1 (``BIRD_NCAND_TEMP``).

    Slot 0 stays on the base client so ``BIRD_NCAND=1`` is bit-for-bit the old
    behaviour.
    """
    return _float("BIRD_NCAND_TEMP", 0.8)


def max_parallel() -> int:
    """Cap on candidate threads (``BIRD_NCAND_PARALLEL``).

    The shared in-flight semaphore in ``llm_invoke`` is what actually protects
    the endpoint; this only bounds thread count.
    """
    return max(1, _int("BIRD_NCAND_PARALLEL", 4))


def schema_slots_enabled() -> bool:
    """Whether slots also vary how they read the schema (``BIRD_SCHEMA_SLOTS``)."""
    return _bool("BIRD_SCHEMA_SLOTS", False)


def pin_strategy() -> str:
    """Force every slot onto one strategy (``BIRD_PIN_STRATEGY``).

    Trades the pool's breadth for repeated sampling of a single generator.
    """
    return (os.getenv("BIRD_PIN_STRATEGY") or "").strip()


def slot_plan() -> tuple[str, ...]:
    """Explicit strategy per slot, comma separated (``BIRD_SLOT_PLAN``).

    Round-robin caps a strategy at its share of the slots; this is the knob
    that moves that share.
    """
    raw = (os.getenv("BIRD_SLOT_PLAN") or "").strip()
    if not raw:
        return ()
    return tuple(tag.strip() for tag in raw.split(",") if tag.strip())


def synthetic_n() -> int:
    """How many same-schema demonstrations to invent (``BIRD_SYNTHETIC_N``)."""
    return max(2, _int("BIRD_SYNTHETIC_N", 3))


def synthetic_reasoning() -> bool:
    """Whether invented demonstrations carry their derivation (``BIRD_SYNTHETIC_REASONING``)."""
    return _bool("BIRD_SYNTHETIC_REASONING", False)


def decomposition_tree() -> bool:
    """Whether decomposition recurses into a tree (``BIRD_DECOMP_TREE``).

    The flat variant yields a linear list of table-access steps, which is
    close enough to the query plan that the two candidates often agree.
    """
    return _bool("BIRD_DECOMP_TREE", False)


def selection_enabled() -> bool:
    """Whether to elect a winner from the pool (``BIRD_SQL_SELECT``).

    Off by default: with it off the pool still rides out on ``sql_candidates``
    for measurement, and slot 0 ships, which is the previous behaviour.
    """
    return _bool("BIRD_SQL_SELECT", False)


def selection_pattern_alpha() -> float:
    """Weight of structural-pattern fit in cluster scoring (``BIRD_SQL_PATTERN_ALPHA``).

    In units of votes, so 1.0 lets a perfect pattern match outweigh one extra
    agreeing candidate. Measured flat across 0.5-3.0 on BIRD dev; 0 disables
    the tie-break and leaves a pure vote.
    """
    return max(0.0, _float("BIRD_SQL_PATTERN_ALPHA", 1.0))


def selection_drop_empty() -> bool:
    """Whether an empty result disqualifies a candidate (``BIRD_SQL_DROP_EMPTY``).

    On by default, and the single highest-value rule in the selector: an
    over-constrained predicate returns no rows, several candidates make that
    same mistake, and the resulting agreement outvotes the one candidate that
    found the data.
    """
    return _bool("BIRD_SQL_DROP_EMPTY", True)
