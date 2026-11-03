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


def is_sql_ensemble_enabled() -> bool:
    """Whether construction routes through the multi-strategy ensemble.

    A third arm rather than a mode of the deep agent, because it is a
    different bet about where the remaining errors are. The deep agent spends
    its budget refining one query; the ensemble spends it writing several and
    letting the database choose. Our candidate-pool measurement says the whole
    remaining headroom sits on the questions where independently written
    queries disagree, which is the ensemble's bet and not the deep agent's.
    Most specific flag wins: this one beats both of the others.
    """
    return os.environ.get("SQL_ENSEMBLE_ENABLED", "").strip().lower() in _TRUTHY


def ensemble_writers_per_method() -> int:
    """Writers per method, so three times this is the writer count.

    Three, matching DataGallery's ``phase1_budget: 3`` across three
    strategies: nine writers, and with the explorer's own query ten
    candidates. Nine is affordable here for the same reason it is there --
    the writers do not explore, they are handed one exploration, so nine of
    them is nine single calls and not nine agents. Against the v10b run that
    is roughly double the spend: the pipeline costs about 9.0 chat completions
    per question today, of which about 3.6 is the deep agent's own loop.

    One detail does not copy over, and :func:`ensemble_writer_temperature`
    handles it. Their three calls per strategy are built from
    ``settings.copy()`` and ``context.copy()`` with no seed and no temperature
    variation, at ``temperature: 0.0`` -- byte-identical requests whose only
    source of difference is their provider not being deterministic. We run at
    temperature 0 too, so relying on the same accident would risk paying for
    nine candidates and getting three. The duplicates here sample instead,
    which makes the diversity deliberate rather than incidental.
    """
    return _int_env("SQL_ENSEMBLE_WRITERS_PER_METHOD", 3)


def ensemble_writer_temperature() -> float:
    """Sampling temperature for the duplicate writers of a method.

    Zero, meaning off: every writer runs on the model exactly as the pipeline
    hands it over, which makes the nine calls byte-identical in threes. That
    is precisely what DataGallery does -- ``settings.copy()``,
    ``context.copy()``, no seed, ``temperature: 0.0`` -- so this reproduces
    their generator rather than an improved version of it.

    Doing it their way first is the informative order. If the duplicates come
    back identical, the provider is deterministic, their three-per-strategy
    buys nothing here, and the arm should be three writers; the run says so
    directly through ``distinct_sql`` in the trajectory. If they come back
    different, the provider varies on its own and the diversity is real
    without anyone paying for it.

    Setting this above zero applies it to the second and later writer of each
    method only, never the first, so the deterministic core of the arm stays
    repeatable. Worth reaching for only if the duplicates turn out identical
    and nine candidates still seem worth having -- CHASE-SQL's position is
    that resampling "often compromises the quality of the candidates" and
    that diversity belongs in the method.
    """
    return _float_env("SQL_ENSEMBLE_WRITER_TEMPERATURE", 0.0)


def ensemble_shuffle_schema() -> bool:
    """Whether each writer reads the schema in its own table/column order.

    The other half of CHASE-SQL's diversity recipe, which pairs raising the
    sampling temperature with shuffling "the order of columns and tables in
    the prompt". Temperature alone was measured over 140 questions: it lifted
    the pool from 1.12 to 1.34 distinct results and recovered 9 of 80
    previously unsolvable questions at no cost to the ones already right, but
    109 of those 140 still came back unanimous. Ordering is what is left to
    vary.

    Off by default so the control stays byte-identical and reproducible.
    """
    return os.environ.get("SQL_ENSEMBLE_SHUFFLE_SCHEMA", "").strip().lower() in _TRUTHY


def ensemble_independent_generators() -> bool:
    """Whether each method explores the database itself instead of sharing one.

    The ensemble's ceiling, not its selection, is what this addresses. With one
    explorer feeding nine writers the pool reaches 1.34 distinct results per
    question and its ceiling reweights to 1110 of 1532 -- the same score the
    single deep agent already gets, because the evidence block tells every
    writer to treat one agent's exploration as fact, and they cannot then
    disagree about the tables it chose or the question it read.

    CHASE-SQL's generators each reason from the schema independently, which is
    where its 82.79% ceiling against a 63.01% single query comes from. This
    reproduces that shape: one deep agent per method, each on its own toolbox,
    in parallel. It costs roughly one deep run per method instead of one per
    question, so it is off by default.
    """
    return os.environ.get("SQL_ENSEMBLE_INDEPENDENT", "").strip().lower() in _TRUTHY


def ensemble_evidence_chars() -> int:
    """Character budget for the exploration handed to the writers.

    The explorer may make up to 40 probe calls and each result can run to 50
    rows, so the raw transcript can be far larger than it is useful to resend
    nine times. This caps it; the renderer drops from the end, keeping the
    orientation calls and losing the late specific checks, which are the ones
    a writer starting fresh can least use.

    Sized so the evidence is comfortably the largest thing in the writer's
    prompt without dominating the context: prompts on the current arm average
    about 3,250 tokens, and this adds up to roughly another 6,000.
    """
    return _int_env("SQL_ENSEMBLE_EVIDENCE_CHARS", 24000)


def ensemble_deep_weight() -> int:
    """How many votes the explorer's own query carries against the writers.

    One, which makes the vote a plain majority exactly as DataGallery's is.

    It is a knob rather than a constant because there is a real argument for
    two, and a reason not to take it yet. Simulated over 1,517 questions
    against the recorded deep-agent scores and the candidate pool:

        cheap candidates  deep weight   floor   ceiling   lost   won
                       2            1     -15       +12     42    27
                       2            2      +0       +27      0     0
                       3            1      +0        +3     41    41
                       4            3     -13       +40     28    15

    At weight 1 the cheap candidates take 42 questions off a deep agent that
    had them right and win 27 back, and with three of them it is 41 against 41
    -- a vote whose information content is indistinguishable from zero. That is
    what the measurement says and it is not a surprise: the cheap path scores
    74.58% where the deep agent scores 76.14%, so two samples of it agreeing
    is not evidence against the better generator.

    At weight 2 a cheap majority can only draw with the deep agent, never
    overrule it, so every disagreement becomes a review rather than a loss --
    a floor of exactly the deep agent's own score with the upside left intact.
    That is a tempting trade and it may yet be the right one here.

    The reason it is not the default: every row of that table compares the
    deep agent against single calls working from the *base* context. The
    writers in this module work from the deep agent's own exploration, which
    is a different and stronger generator, and nothing measures them yet.
    Weighting against them would be importing a result from a setup that no
    longer exists. Raise it to 2 if the first run shows the writers outvoting
    the explorer and losing.
    """
    return _int_env("SQL_ENSEMBLE_DEEP_WEIGHT", 1)


def ensemble_parallelism() -> int:
    """Concurrent generation calls inside one question.

    The harness already runs several questions at once, so this multiplies
    rather than replaces that concurrency. Held at the width of one round
    because a rate-limited call returns no SQL, and no SQL scores exactly like
    a wrong answer.
    """
    return _int_env("SQL_ENSEMBLE_PARALLELISM", 3)


def ensemble_review_threshold() -> float:
    """Share of the vote a leading result needs before the model is skipped.

    The gate is narrow on measurement. Our own judges scored 184 and 186 on
    the contested questions against 191 for simply taking the first candidate,
    and CHASE-SQL independently puts an untuned model at 58% accuracy on this
    comparison. A judge barely better than a coin toss should only be asked
    when the coin is genuinely in the air, which is what a tie or a leader
    under half the votes means.
    """
    return _float_env("SQL_ENSEMBLE_REVIEW_THRESHOLD", 0.5)


def ensemble_max_rows() -> int:
    """Row cap for a candidate's execution, and so the resolution of the vote.

    Far above the probe layer's 50. Probes answer questions about the data;
    these are whole answers being compared against each other, and two results
    that differ only past the cap are indistinguishable to the clustering --
    which would merge a right answer into a wrong one's cluster and hand it
    the vote.
    """
    return _int_env("SQL_ENSEMBLE_MAX_ROWS", 10000)


def ensemble_max_executions() -> int:
    """Database calls the ensemble's own executor may make.

    Sized above the largest possible pool so the budget cannot run out partway
    through a round, which would silently drop the last strategies from the
    vote and bias it toward whichever ran first.
    """
    return _int_env("SQL_ENSEMBLE_MAX_EXECUTIONS", 40)


def ensemble_deadline_s() -> float:
    """Wall-clock budget for the whole ensemble, in seconds.

    Its own knob rather than :func:`deadline_s`, and much larger. That budget
    was sized for one agent taking turns; three parallel generations can spend
    most of it before the first round is even scored, and an ensemble that
    runs out of time before escalating is just a slower single call. Checked
    only at the escalation decision, so a question that is already over
    budget answers from the round it has instead of abandoning it.
    """
    return _float_env("SQL_ENSEMBLE_DEADLINE_S", 600.0)


def ensemble_timeout_s() -> float:
    """Per-candidate wall-clock cap.

    Longer than a probe's five seconds because a candidate is a full answer
    over full tables, and shorter than the question's own budget because one
    runaway query must not spend the vote's entire time.
    """
    return _float_env("SQL_ENSEMBLE_TIMEOUT_S", 30.0)


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


def ensemble_candidates_path() -> str:
    """File to append every candidate to, or "" to keep none.

    The vote discards nine of ten queries, and once discarded there is no way
    to ask the only question that matters about a selector -- whether the
    answer it threw away was the right one. Off by default because a full run
    writes 1,534 of these; set it for any run whose selection is being judged.
    """
    return os.environ.get("SQL_ENSEMBLE_CANDIDATES_PATH", "").strip()


__all__ = [
    "is_sql_agent_enabled",
    "is_sql_ensemble_enabled",
    "max_turns",
    "max_submission_bounces",
    "max_probe_calls",
    "max_sql_calls",
    "deadline_s",
    "max_rows_shown",
    "max_cell_chars",
    "ensemble_writers_per_method",
    "ensemble_writer_temperature",
    "ensemble_shuffle_schema",
    "ensemble_independent_generators",
    "ensemble_evidence_chars",
    "ensemble_deep_weight",
    "ensemble_parallelism",
    "ensemble_review_threshold",
    "ensemble_max_rows",
    "ensemble_max_executions",
    "ensemble_deadline_s",
    "ensemble_timeout_s",
    "deep_protocol_order",
    "ensemble_candidates_path",
    "reflector_enabled",
    "reflector_max_faults",
]
