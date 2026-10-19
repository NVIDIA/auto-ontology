# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the agentic SQL construction loop.

Construction normally costs one LLM call. The agent trades that for a bounded
conversation in which it can read the database before committing, so every
knob here exists to put a ceiling on that trade. All are environment-driven,
and the feature itself is off unless ``SQL_AGENT_ENABLED`` says otherwise.
"""

from __future__ import annotations

import os

_TRUTHY = {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


def is_sql_agent_enabled() -> bool:
    """Whether construction routes through the tool-calling loop.

    Off by default: an unconfigured run keeps the single-call
    ``construct_sql_from_candidates`` path, so the agent can be A/B'd against
    it without a second branch.
    """
    return os.environ.get("SQL_AGENT_ENABLED", "").strip().lower() in _TRUTHY


def is_sql_deep_agent_enabled() -> bool:
    """Whether construction routes through the deep agent.

    Separate from ``SQL_AGENT_ENABLED`` because the two are different bets and
    have to be scored apart. The loop refines one reading of the question; the
    deep agent is there to hold several and decide between them, which is the
    only thing the candidate-pool measurements left room for. When both flags
    are set this one wins, since it is the more specific request.
    """
    return os.environ.get("SQL_DEEP_AGENT_ENABLED", "").strip().lower() in _TRUTHY


def is_deep_adjudicator_enabled() -> bool:
    """Whether the deep agent may consult the reading adjudicator.

    Off by default on measurement, not on principle. Over 259 paired questions
    the arm was -3 on the 21 questions where the adjudicator was consulted and
    -5 on the other 238 — roughly seven times worse per question — and on four
    of them the agent's own final thought credited the adjudication for a
    filter the question never asked for. It runs isolated, so it never sees the
    protocol the main agent works by, which left it more confident and less
    informed than the agent it was meant to check. Kept behind a flag because
    the idea may be right and the framing wrong.
    """
    return os.environ.get("SQL_DEEP_ADJUDICATOR", "").strip().lower() in _TRUTHY


def deep_recursion_limit() -> int:
    """LangGraph steps the deep agent may take, tools and sub-agents included.

    Not the same currency as :func:`max_turns`: one model turn with three tool
    calls spends several steps here. Sized so a run can enumerate readings,
    probe each, consult the adjudicator and still have room to revise, and low
    enough that a question which starts looping dies rather than eating the
    deadline alone.
    """
    return _int_env("SQL_DEEP_AGENT_RECURSION_LIMIT", 60)


def max_turns() -> int:
    """LLM turns the loop may take before it must answer with what it has.

    Sized for a question that explores, drafts, gets its submission bounced by
    the gate, and fixes it — twice over.
    """
    return _int_env("SQL_AGENT_MAX_TURNS", 12)


def max_submission_bounces() -> int:
    """How many times the gate may send a submission back for repair.

    Bounded because the gate's checks are heuristics: two of them disagreeing
    with a query the model is sure about should cost a revision, not the
    question. After this many the submission is taken as-is.
    """
    return _int_env("SQL_AGENT_MAX_BOUNCES", 2)


def max_probe_calls() -> int:
    """Database calls the whole construction may make.

    Above the deterministic probe layer's own default: that budget was sized
    for a handful of targeted checks on one finished query, where this covers
    the model's exploration plus a full diagnostic pass on each submission.
    """
    return _int_env("SQL_AGENT_MAX_PROBE_CALLS", 80)


def max_sql_calls() -> int:
    """``run_sql`` calls allowed per question.

    Sits under the probe executor's own ``DB_PROBE_MAX_CALLS`` budget, which
    the diagnostic tools also draw from, so this caps the share the model can
    spend on free exploration.
    """
    return _int_env("SQL_AGENT_MAX_SQL", 12)


def deadline_s() -> float:
    """Wall-clock budget for the whole loop, in seconds.

    On expiry the loop stops and returns the best SQL it has seen rather than
    failing, so a slow question degrades to a worse answer and not to no
    answer.
    """
    return _float_env("SQL_AGENT_DEADLINE_S", 120.0)


def max_rows_shown() -> int:
    """Rows from a tool result that get rendered back into the conversation."""
    return _int_env("SQL_AGENT_MAX_ROWS_SHOWN", 20)


def max_cell_chars() -> int:
    """Per-cell truncation for rendered rows, so one wide column cannot flood
    the context window."""
    return _int_env("SQL_AGENT_MAX_CELL_CHARS", 200)


def reflector_enabled() -> bool:
    """Whether a second reader reviews and may repair the accepted query.

    Off by default. It costs one extra call on every question and a second on
    the questions it finds fault with, and it is the kind of change that can
    only be justified by a paired measurement rather than by argument.
    """
    return os.environ.get("SQL_REFLECTOR_ENABLED", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def reflector_max_faults() -> int:
    """How many faults one review may raise.

    Capped low on purpose. A review listing six things wrong with a query is
    not a review, it is a rewrite with extra steps, and the repair it produces
    stops being minimal -- which is the only property making the repair safer
    than regenerating from scratch.
    """
    try:
        return max(1, int(os.environ.get("SQL_REFLECTOR_MAX_FAULTS", "2")))
    except ValueError:
        return 2


def deep_protocol_order() -> tuple[str, ...]:
    """Order of the deep agent's protocol blocks, or () for the written order.

    Set to a comma-separated list of block names from
    ``deep._PROTOCOL_BLOCKS``. Exists so that a search over block orderings can
    vary the prompt per run without editing the prompt, which keeps the wording
    provably identical across the arms of that search: every arm is assembled
    from the same strings.
    """
    raw = os.environ.get("SQL_DEEP_PROTOCOL_ORDER", "")
    return tuple(part.strip() for part in raw.split(",") if part.strip())


__all__ = [
    "is_sql_agent_enabled",
    "max_turns",
    "max_submission_bounces",
    "max_probe_calls",
    "max_sql_calls",
    "deadline_s",
    "max_rows_shown",
    "max_cell_chars",
    "deep_protocol_order",
    "reflector_enabled",
    "reflector_max_faults",
]
