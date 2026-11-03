# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Construct SQL by writing the query several ways and picking by execution.

Both published systems near the top of the BIRD leaderboard are built the same
way, and it is not the way we build ours. CHASE-SQL (Google) and
Agentar-Scale-SQL (Ant Group) each generate candidates from *structurally
different* reasoning strategies and then select between them; Agentar calls its
version "Parallel Scaling using Diverse Synthesis and Tournament Selection".
DataGallery's BIRD agent, currently first on the dev split, does the same with
three strategies named ``dc``, ``skeleton`` and ``icl``.

The measurement that says this is worth building here is ours, not theirs. On a
pool of six candidates per question over 1,517 BIRD dev questions:

    first candidate        1122   73.96%
    majority cluster       1131   74.56%
    best available         1211   79.83%

and that headroom is not spread out. Split by whether the candidates actually
disagree:

    all six agree          1128 questions, we get 82.5%, ceiling 82.5%
    they disagree           389 questions, we get 49.1%, ceiling 72.0%

So the entire opportunity -- 89 questions -- sits in the quarter of the set
where the candidates disagree, and on the three quarters where they agree there
is nothing to win. That shape is what this module is built around: agreement is
treated as the stopping condition, and effort is spent only where there is
disagreement to resolve.

Two things that pool got wrong and this does not. It was six samples of one
pipeline at temperature, so 1,128 of 1,517 questions came out unanimous and the
errors were shared -- CHASE-SQL is explicit that resampling "often compromises
the quality of the candidates" and that diversity has to come from the method
instead. And it judged every disagreement with an untuned model, which cost
more than it earned: our judges scored 184 and 186 against 191 for simply
taking the first candidate. CHASE-SQL measured the same thing and reported an
off-the-shelf model at 58% accuracy on this comparison, barely better than a
coin toss. The gate here is therefore deliberately narrow, as DataGallery's is:
the vote decides unless it is tied or short of a majority.

How nine candidates are affordable is the part worth reading twice, because it
is the part we had wrong. The obvious way to get nine is to run nine agents,
and nine agents is a run nobody will wait for. DataGallery does not do that.
Its ``perceptor`` narrows the schema, retrieves candidate values and samples
rows *once*, and then nine single LLM calls write SQL from that one prepared
context, fired concurrently. The looking is paid once and amortised across
every candidate; only the writing is paid nine times.

This module does the same thing with the deep agent in the perceptor's place.
Phase one runs the deep agent once: it explores, and what it learns -- every
tool call it made and every result that came back -- is kept. Phase two hands
that evidence to nine writers, three each under divide-and-conquer,
skeleton-first and pattern-transfer, as single calls. The deep agent's own
query joins them as a tenth candidate, already paid for.

Measured against the v10b run, the arithmetic that makes this work: the whole
pipeline spends about 9.0 chat completions per question, of which roughly 3.6
are the deep agent's own loop and the rest is the graph around it. Nine
writers add about nine calls to a question that already costs nine, so this is
near twice the spend -- where nine agents would have been about four times it,
and three independent agents about 1.8 times.

What is given up is independence. Three separate agents explore separately, so
a wrong table found by one does not reach the others; nine writers sharing one
exploration all inherit whatever that exploration got wrong. The winners
accept exactly that exposure and sit at 78.10 on this split, which is the best
evidence available that it is the right trade.

One correction worth recording, since it shaped an earlier version of this
file. Our single call scores 74.58% against the deep agent's 76.14%, and a
simulated vote mixing those two lost 42 questions to win 27. That measurement
is about single calls working from the *base* context, and it does not
transfer to writers working from the deep agent's gathered evidence -- which
is precisely why DataGallery's single calls do well. The writers here are
therefore equals and the vote among them is a plain majority, as theirs is.
:func:`config.ensemble_deep_weight` survives for the cheap mixed arm, where
the asymmetry is real.

What this does *not* do is train a selector, which is what both winners do and
what would be needed to reach for the full 89. Without that, the honest target
is the part a vote can reach.
"""

from __future__ import annotations

import json
import logging
import random
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from langchain_core.messages import BaseMessage, HumanMessage

from auto_ontology.retrieval.text_to_sql.db_probe.executor import ProbeExecutor
from auto_ontology.retrieval.text_to_sql.models import SQLGenerationModel
from auto_ontology.retrieval.text_to_sql.sql_agent import config
from auto_ontology.retrieval.text_to_sql.sql_agent.loop import SQLAgentOutcome
from auto_ontology.retrieval.text_to_sql.sql_agent.tools import SQLAgentToolbox
from auto_ontology.utils.llm_invoke import safe_invoke_with_structured_output

logger = logging.getLogger(__name__)

#: Serialises appends to the candidate record. The harness runs many questions
#: at once in one process, and two workers writing a line each at the same time
#: produce one unparseable line rather than two good ones.
_RECORD_LOCK = threading.Lock()

#: The two roles a candidate can be produced in. The ``deep`` explorer is the
#: full tool-calling agent: it is the only thing here that touches the
#: database, and it runs once. A ``writer`` is one LLM call that cannot
#: explore and does not need to, because it is handed what the explorer found.
#:
#: This split is the whole cost argument. Nine explorers is a run nobody will
#: wait for; nine writers over one exploration is what DataGallery actually
#: does, with its ``perceptor`` in the explorer's place.
DEEP_ENGINE = "deep"
WRITER_ENGINE = "writer"

#: The ways of arriving at a query, in the order DataGallery runs them. A
#: method is a way of thinking and says nothing about who applies it.
METHODS = ("dc", "skeleton", "icl")

#: Per-strategy method guidance, appended to the shared construction prompt.
#:
#: Only the *method* differs. Every rule about what a correct query looks like
#: stays in the shared prompt where it already lives, because three copies of
#: those rules would drift apart and then the arm would be measuring the drift
#: rather than the diversity. These say how to think, not what is true.
_METHOD: dict[str, str] = {
    "dc": """\
Work this question out by divide and conquer.

First break it into the smallest information needs it contains, and write a
pseudo-query for each one, leaving a placeholder where a sub-question is not
yet answered. Then resolve each sub-question into real SQL on its own. Then
reassemble from the bottom up, replacing each placeholder with the SQL that
answers it.

The point of the decomposition is that a question with three conditions in it
is three decisions, and taking them one at a time is what stops the third from
being forgotten. Do not collapse the steps.""",
    "skeleton": """\
Work this question out by building the query's skeleton before its contents.

First decide which clauses the answer needs at all -- what is selected, what is
joined, what is filtered, whether there is grouping, whether there is ordering,
whether there is a limit, whether a subquery is required -- and write the
skeleton with placeholders where the names go. Then fill the placeholders with
the exact names from the schema.

Deciding the shape first is what stops a missing GROUP BY or a missing HAVING
from being noticed only after the names are already chosen.""",
    "icl": """\
Work this question out from the pattern it belongs to.

Look at the reference queries you were given. Identify which of them answers a
question shaped like this one -- the same kind of aggregation, the same kind of
filter, the same use of a subquery or a limit -- and follow that structure,
substituting this schema's tables and columns for theirs.

The reference queries come from other databases, so the names in them mean
nothing here. What transfers is the shape. Where no reference matches, say so
and write the query directly rather than forcing a pattern onto it.""",
}

#: Shown to the selector when the vote cannot decide. Kept short on purpose:
#: the thing being judged is which result answers the question, and the model's
#: attention is better spent on the rows than on instructions about rows.
_REVIEW_SYSTEM = """\
You are choosing which of several SQL queries answers a question correctly.

Each candidate has already run. You are given its SQL, the columns it returned,
a preview of its rows, and how many of the independently written queries
produced that same result.

Judge the result against the question, not the SQL against your taste. Decide
first how many values an answer row should contain and what they should be,
then pick the candidate whose result has that shape and those values. A
candidate returning extra columns the question never asked for is wrong, and so
is one missing a column the question named.

The agreement count is weak evidence and nothing more. Several queries written
the same way make the same mistake together, so a result with more agreement is
not for that reason the right one.

Reply with the id of the single best candidate and one sentence of reason."""


@dataclass(frozen=True)
class _Spec:
    """Who produced a candidate: an engine, and the method it applied.

    ``order`` is the tie-break of last resort. The explorer is given order 0,
    so a drawn vote that the review declines to settle falls to its query --
    the one generator here with a measured score behind it.
    """

    engine: str
    method: Optional[str]
    order: int
    #: True for the second and later writer of a method. Those have to sample
    #: to be worth running at all, since at temperature 0 they would return
    #: what the first one returned.
    replica: bool = False
    #: Which writer of this method, from 0. Carried because two replicas of a
    #: method were previously indistinguishable: both reported as
    #: ``writer:dc~``, so an audit of the pool counted two generators as one
    #: and could not tell which of them produced a query. It is also the seed
    #: for this writer's schema ordering, which is what makes the copies
    #: differ by something other than sampling luck.
    copy: int = 0

    @property
    def name(self) -> str:
        base = f"{self.engine}:{self.method}" if self.method else self.engine
        return f"{base}~{self.copy}" if self.replica else base


@dataclass
class _Candidate:
    """One query, and what happened when it ran."""

    sql: str
    thought: str
    #: The generator's user-facing summary. Carried because the node returns a
    #: ``SQLGenerationModel``, whose ``response`` may not be empty -- and the
    #: summary that belongs to the chosen query is the only honest one to give.
    response: str
    spec: _Spec
    phase: int
    #: ``None`` when the query failed; otherwise the result as an unordered set
    #: of row tuples, which is the same comparison the official metric makes.
    rows: Optional[frozenset] = None
    columns: tuple[str, ...] = ()
    row_count: int = 0
    truncated: bool = False
    error: Optional[str] = None
    preview: list = field(default_factory=list)
    #: Votes this candidate carries. Assigned at clustering time, because it
    #: depends on how many other candidates there are.
    weight: int = 1

    @property
    def ran(self) -> bool:
        return self.error is None and self.rows is not None

    @property
    def is_deep(self) -> bool:
        return self.spec.engine == DEEP_ENGINE


@dataclass
class _Group:
    """Candidates whose queries produced the same result."""

    key: frozenset
    members: list[_Candidate]

    @property
    def votes(self) -> int:
        """Weighted votes, not candidates.

        Equal weights when the candidates are equals, which is the point of
        running one engine under three methods. The weighting only does
        anything in a mixed pool -- see :func:`_assign_weights`.
        """
        return sum(m.weight for m in self.members)

    @property
    def rank(self) -> int:
        """Where this result sits in the configured order, for breaking ties.

        The earliest-configured strategy *present* in the group, not the
        strategy of its representative, so a draw goes to the cluster the
        strongest generator is in.
        """
        return min(m.spec.order for m in self.members)

    @property
    def best(self) -> _Candidate:
        """The member to answer with.

        Every member returned identical rows, so this cannot change the score
        -- it decides which SQL and which reasoning travel downstream. The
        strongest strategy first, because its ``thought`` is the one the
        reconstruction loop will have to work from if the query still needs
        repair; then the shortest, as the least likely to carry a join it
        never used.
        """
        return min(self.members, key=lambda c: (c.spec.order, len(c.sql)))


_EVIDENCE_HEADER = """\
Before you answer, here is what has already been checked against this database.
Another agent explored it and these are the calls it made and what came back.
Treat these results as fact: they are what the database actually returned, and
they are more reliable than the schema summary alone. Do not restate them, use
them."""


def _evidence_block(toolbox: SQLAgentToolbox, budget: int) -> str:
    """Render the exploration as text the writers can read. ``""`` if nothing.

    This is the whole reason nine candidates are affordable: it is the
    exploration, paid for once, in a form that can be handed to a generator
    that cannot explore. It stands in for DataGallery's prepared context.

    Truncated from the end rather than the start. The agent's early calls are
    its orientation -- what tables exist, what a row looks like -- and its
    later ones are increasingly specific checks on a query it has already
    mostly decided. If something has to be dropped, the specific checks are
    the ones a writer starting fresh can least use.
    """
    if not toolbox.calls:
        return ""
    parts: list[str] = []
    spent = 0
    for call in toolbox.calls:
        args = ", ".join(f"{k}={v!r}" for k, v in (call.get("args") or {}).items())
        result = str(call.get("result") or "")
        rendered = f"$ {call.get('tool')}({args})\n{result}"
        if spent + len(rendered) > budget:
            parts.append(f"[{len(toolbox.calls) - len(parts)} further call(s) omitted]")
            break
        parts.append(rendered)
        spent += len(rendered)
    return f"{_EVIDENCE_HEADER}\n\n" + "\n\n".join(parts)


_TABLE_HEAD = "TABLE: "
_COLUMN_LINE = "    - "


def _shuffle_schema_text(text: str, seed: int) -> str:
    """Reorder the ``TABLE:`` blocks, and the columns inside each, by *seed*.

    Works on the rendered text rather than the catalog because the schema is
    rendered once, upstream, before the writers exist; re-deriving it per
    writer would mean re-running retrieval. ``format_tables_for_prompt``
    renders one table per blank-line-separated block and one column per line,
    which is what makes the text safe to permute.

    Continuation lines are kept with the column above them. A description
    carrying a newline would otherwise be detached from its column and left
    describing whichever column the permutation happened to put there.
    """
    blocks = text.split("\n\n")
    rng = random.Random(seed)

    def one_block(block: str) -> str:
        lines = block.split("\n")
        head: list[str] = []
        cols: list[list[str]] = []
        for line in lines:
            if line.startswith(_COLUMN_LINE):
                cols.append([line])
            elif cols:
                cols[-1].append(line)
            else:
                head.append(line)
        if len(cols) < 2:
            return block
        rng.shuffle(cols)
        return "\n".join(head + [ln for col in cols for ln in col])

    tables = [b for b in blocks if b.lstrip().startswith(_TABLE_HEAD)]
    if len(tables) < 2:
        return "\n\n".join(one_block(b) for b in blocks)

    shuffled = [one_block(b) for b in tables]
    rng.shuffle(shuffled)
    # Non-table blocks hold their position: they are the instructions around
    # the schema, and moving those changes what the writer is asked to do
    # rather than the order it reads the tables in.
    out, nxt = [], iter(shuffled)
    for block in blocks:
        out.append(next(nxt) if block.lstrip().startswith(_TABLE_HEAD) else block)
    return "\n\n".join(out)


def _reordered(messages: list[BaseMessage], seed: int) -> list[BaseMessage]:
    """Copy *messages* with every schema block permuted for this writer."""
    out: list[BaseMessage] = []
    for message in messages:
        content = getattr(message, "content", None)
        if isinstance(content, str) and _TABLE_HEAD in content:
            out.append(message.model_copy(update={"content": _shuffle_schema_text(content, seed)}))
        else:
            out.append(message)
    return out


def _writer_messages(
    messages: list[BaseMessage], method: Optional[str], evidence: str
) -> list[BaseMessage]:
    """The shared prompt, the exploration, then one method note.

    All three appended as trailing turns rather than edited into the system
    message, so every arm starts from a byte-identical prompt and an A/B
    between them tests the ensemble rather than a rewording.

    Order matters: the evidence is what the writer knows, the method is what
    it should do with it, and the instruction lands last so it is the thing
    the model is holding when it starts to answer.
    """
    out = list(messages)
    if evidence:
        out.append(HumanMessage(content=evidence))
    if method:
        out.append(HumanMessage(content=_METHOD[method]))
    return out


def _explore(
    llm: Any, messages: list[BaseMessage], toolbox: SQLAgentToolbox, state: Any
) -> Optional[_Candidate]:
    """Run the deep agent once. Its query is a candidate; its calls are the evidence.

    On the shared toolbox deliberately, not a fresh one: this is the only
    agent that runs, so there is nothing to race it, and the tables it
    discovers have to land where the node will look for them.
    """
    # Imported here rather than at module scope: the deep agent imports the
    # toolbox, which imports this package, and at module scope that closes
    # into a cycle.
    from auto_ontology.retrieval.text_to_sql.sql_agent.deep import run_deep_sql_agent

    spec = _Spec(DEEP_ENGINE, None, 0)
    try:
        result = run_deep_sql_agent(llm, messages, toolbox, state).result
    except Exception:  # noqa: BLE001
        # The explorer failing costs its candidate, not the question: whatever
        # it managed to look up before it broke is still on the toolbox, and
        # the writers can still work from that.
        logger.warning("Ensemble explorer failed", exc_info=True)
        return None
    sql = (getattr(result, "sql_code", "") or "").strip()
    if not sql:
        return None
    return _Candidate(
        sql=sql,
        thought=(getattr(result, "thought", "") or "").strip(),
        response=(getattr(result, "response", "") or "").strip(),
        spec=spec,
        phase=1,
    )


def _explore_independently(
    llm: Any, messages: list[BaseMessage], state: Any, shared: SQLAgentToolbox
) -> list[_Candidate]:
    """One deep agent per method, each exploring on a toolbox of its own.

    The point is that nothing is shared downstream of the question. Each agent
    reads the schema in its own order, runs its own probes, and reaches its own
    reading, so two of them disagreeing is evidence rather than paraphrase --
    which is the property the shared-evidence arrangement cannot have at any
    temperature.

    The first method reuses the caller's toolbox so the node still gets the
    discovered tables it has to return for validation; the rest get their own
    and are closed here. A method whose agent fails costs its candidate only.
    """
    from auto_ontology.retrieval.text_to_sql.sql_agent.deep import run_deep_sql_agent

    methods = list(METHODS)
    owned: list[SQLAgentToolbox] = []

    def one(indexed: tuple[int, str]) -> Optional[_Candidate]:
        order, method = indexed
        box = shared
        if order > 1:
            try:
                box = SQLAgentToolbox.from_state(state)
            except Exception:  # noqa: BLE001
                logger.warning("Could not build a toolbox for %s", method, exc_info=True)
                return None
            owned.append(box)
        prompt = _writer_messages(
            _reordered(messages, zlib.crc32(method.encode()))
            if config.ensemble_shuffle_schema()
            else messages,
            method,
            "",
        )
        try:
            result = run_deep_sql_agent(llm, prompt, box, state).result
        except Exception:  # noqa: BLE001
            logger.warning("Independent generator %s failed", method, exc_info=True)
            return None
        sql = (getattr(result, "sql_code", "") or "").strip()
        if not sql:
            return None
        return _Candidate(
            sql=sql,
            thought=(getattr(result, "thought", "") or "").strip(),
            response=(getattr(result, "response", "") or "").strip(),
            spec=_Spec(DEEP_ENGINE, method, order),
            phase=1,
        )

    width = max(1, min(config.ensemble_parallelism(), len(methods)))
    try:
        with ThreadPoolExecutor(max_workers=width) as pool:
            produced = list(pool.map(one, enumerate(methods, start=1)))
    finally:
        for box in owned:
            try:
                box.close()
            except Exception:  # noqa: BLE001
                logger.warning("Could not close a generator toolbox", exc_info=True)
    return [c for c in produced if c is not None]


def _sampling(llm: Any, temperature: float) -> Any:
    """The same model set to sample. Falls back to the model as given.

    Needed because a duplicate writer is only a candidate if something makes
    it differ. At the pipeline's temperature of 0 a second writer with the
    same method and the same evidence returns the same SQL, which is a
    duplicate vote bought at full price.
    """
    try:
        return llm.bind(temperature=temperature)
    except Exception:  # noqa: BLE001
        logger.warning(
            "Could not set temperature %.2f on the model; the duplicate "
            "writers will repeat the first one",
            temperature,
        )
        return llm


def _write(
    llm: Any, messages: list[BaseMessage], specs: list[_Spec], evidence: str
) -> list[_Candidate]:
    """Fire every writer over the shared evidence, in parallel up to the cap.

    The cap exists because the harness already runs many questions at once;
    multiplying that by the writers without a limit turns a working run into a
    throttled one, and a throttled call returns no SQL, which scores exactly
    like a wrong answer.

    By default nothing samples: every writer gets the model as handed over,
    so the three calls of a method are byte-identical and differ only if the
    provider is not deterministic. That is DataGallery's arrangement exactly,
    and running it first is what tells us whether their three-per-strategy
    transfers -- ``distinct_sql`` in the trajectory is the answer.
    """
    if not specs:
        return []
    temperature = config.ensemble_writer_temperature()
    sampler: Optional[Any] = None
    if temperature > 0 and any(s.replica for s in specs):
        sampler = _sampling(llm, temperature)

    shuffle = config.ensemble_shuffle_schema()

    def one(spec: _Spec) -> Optional[_Candidate]:
        model = sampler if (spec.replica and sampler is not None) else llm
        # The first writer of each method keeps the catalog's own order, so
        # the arm never loses the ordering the rest of the pipeline was tuned
        # against; only the replicas read it rearranged.
        prompt = messages
        if shuffle and spec.replica:
            # crc32 rather than hash(): string hashing is salted per process,
            # so hash() would give a writer a different schema order on every
            # run and make a rerun of this arm unreproducible.
            prompt = _reordered(messages, zlib.crc32(spec.name.encode()))
        try:
            result = safe_invoke_with_structured_output(
                model,
                _writer_messages(prompt, spec.method, evidence),
                SQLGenerationModel,
            )
        except Exception:  # noqa: BLE001
            # One writer failing is not the question failing. The others are
            # still in flight and a vote of two is still a vote.
            logger.warning("Writer %s failed", spec.name, exc_info=True)
            return None
        sql = (getattr(result, "sql_code", "") or "").strip()
        if not sql:
            return None
        return _Candidate(
            sql=sql,
            thought=(getattr(result, "thought", "") or "").strip(),
            response=(getattr(result, "response", "") or "").strip(),
            spec=spec,
            phase=2,
        )

    width = max(1, min(config.ensemble_parallelism(), len(specs)))
    with ThreadPoolExecutor(max_workers=width) as pool:
        produced = list(pool.map(one, specs))
    return [c for c in produced if c is not None]


def _row_key(row: Any) -> tuple:
    """One result row as a hashable tuple of its values.

    ``ProbeExecutor`` hands back ``to_dict(orient="records")``, so a row is a
    mapping and iterating it yields *column names*. Keying on those would make
    every query with the same column list look like the same result, which is
    exactly the failure the clustering exists to avoid. Values are stringified
    because a probe can return a list or a dict in a JSON column, and an
    unhashable value would take the question down with it.
    """
    values = row.values() if isinstance(row, dict) else row
    out = []
    for value in values:
        try:
            hash(value)
            out.append(value)
        except TypeError:
            out.append(repr(value))
    return tuple(out)


def _execute(candidates: list[_Candidate], executor: ProbeExecutor, cap: int) -> None:
    """Run each candidate and record its result on it, in place.

    Deduplicated by SQL text first: identical queries cannot produce different
    results, and the execution budget is better spent elsewhere. The duplicates
    still count toward the vote, which is the point of counting votes.
    """
    seen: dict[str, _Candidate] = {}
    for cand in candidates:
        key = " ".join(cand.sql.split()).lower()
        done = seen.get(key)
        if done is not None:
            cand.rows, cand.columns = done.rows, done.columns
            cand.row_count, cand.truncated = done.row_count, done.truncated
            cand.error, cand.preview = done.error, done.preview
            continue
        entry = executor.run(cand.sql, purpose="ensemble candidate")
        if not entry.get("ok"):
            cand.error = str(entry.get("error") or "did not run")
        else:
            rows = entry.get("rows") or []
            cand.rows = frozenset(_row_key(r) for r in rows)
            cand.row_count = int(entry.get("row_count") or len(rows))
            # The executor appends ``LIMIT cap`` to any query without one, so
            # its own ``truncated`` flag can never fire -- the row count it
            # compares against the cap was already clipped by it. A result
            # sitting exactly on the cap is therefore the only evidence of
            # truncation available, and has to be read as such.
            cand.truncated = bool(entry.get("truncated")) or cand.row_count >= cap
            cand.columns = tuple(rows[0].keys()) if rows and isinstance(rows[0], dict) else ()
            cand.preview = [_row_key(r) for r in rows[:5]]
        seen[key] = cand


def _assign_weights(candidates: list[_Candidate]) -> None:
    """How many votes each candidate carries. One each, by default.

    A plain majority, as DataGallery's is, and for the same reason: the
    writers all see the same evidence and differ only in method, so they are
    equals and counting them is meaningful.

    The explorer is the one candidate that is not quite their equal -- it had
    tools and they did not -- and :func:`config.ensemble_deep_weight` can give
    it extra votes. It defaults to 1 because the measurement that argued for
    weighting does not apply here: that simulation compared the deep agent
    against single calls working from the *base* context, where it beat them
    76.14% to 74.58%. Writers working from its own exploration are a different
    and stronger generator, and nothing yet measures them. Left at 1 until
    something does.
    """
    weight = max(1, config.ensemble_deep_weight())
    for cand in candidates:
        cand.weight = weight if cand.is_deep else 1


def _cluster(candidates: list[_Candidate]) -> list[_Group]:
    """Group the candidates that ran by the result they produced.

    Queries that failed are dropped: a candidate with no result cannot win a
    vote about results. If any candidate returned rows, the ones that returned
    none are dropped too -- an empty result is almost always a filter that
    matched nothing rather than the true answer, and letting empties vote lets
    several identically-wrong filters outvote one correct query.

    Truncation is part of the key. Two results that agree up to the row cap are
    not known to agree, and treating them as one cluster would invent an
    agreement the database never confirmed.

    Column *names* are deliberately not part of the key, because they are not
    part of the metric: BIRD compares ``set(cursor.fetchall())`` against
    ``set(cursor.fetchall())``, which carries values and no headers. Two
    queries that differ only in how they alias a column are the same answer
    and must land in the same cluster or they split their own vote.
    """
    ran = [c for c in candidates if c.ran]
    if not ran:
        return []
    non_empty = [c for c in ran if c.row_count]
    pool = non_empty or ran
    _assign_weights(pool)
    groups: dict[tuple, _Group] = {}
    for cand in pool:
        key = (cand.rows, cand.truncated)
        group = groups.get(key)
        if group is None:
            groups[key] = _Group(key=cand.rows, members=[cand])
        else:
            group.members.append(cand)
    # Largest vote first, then the strategy order, so the ordering is fixed and
    # a rerun of the same candidates selects the same way.
    return sorted(groups.values(), key=lambda g: (-g.votes, g.rank))


def _is_decided(groups: list[_Group], threshold: float) -> bool:
    """Whether the vote settles it without asking a model.

    Two conditions, both from DataGallery's selector: a clear winner must be
    strictly ahead of the runner-up, and must hold at least ``threshold`` of
    the votes cast. Anything else is a disagreement worth a second opinion.
    """
    if not groups:
        return False
    if len(groups) == 1:
        return True
    cast = sum(g.votes for g in groups)
    return groups[0].votes > groups[1].votes and groups[0].votes / cast >= threshold


def _review(llm: Any, groups: list[_Group], question: str) -> Optional[_Group]:
    """Ask a model which result answers the question. ``None`` if it cannot.

    Deliberately given one representative per result rather than every
    candidate: candidates inside a group are interchangeable by construction,
    and showing six near-identical queries is what makes this comparison hard
    for an untuned model in the first place.
    """
    cast = sum(g.votes for g in groups)
    lines = []
    for idx, group in enumerate(groups):
        cand = group.best
        lines.append(
            f"[{idx}] agreed by {group.votes} of {cast} queries"
            f"\n  sql: {cand.sql}"
            f"\n  columns: {list(cand.columns) or 'unknown'}"
            f"\n  rows returned: {cand.row_count}"
            f"{' (truncated)' if cand.truncated else ''}"
            f"\n  first rows: {cand.preview}"
        )
    prompt = (
        f"{_REVIEW_SYSTEM}\n\nQuestion: {question}\n\n"
        + "\n\n".join(lines)
        + f"\n\nReply with sql_code set to the chosen candidate's id "
        f"(a number from 0 to {len(groups) - 1}) and thought set to the reason."
    )
    try:
        verdict = safe_invoke_with_structured_output(
            llm, [HumanMessage(content=prompt)], SQLGenerationModel
        )
    except Exception:  # noqa: BLE001
        logger.warning("Ensemble review failed; falling back to the vote", exc_info=True)
        return None
    raw = (getattr(verdict, "sql_code", "") or "").strip()
    digits = "".join(ch for ch in raw if ch.isdigit())
    if not digits:
        return None
    pick = int(digits)
    if not 0 <= pick < len(groups):
        return None
    logger.info(
        "Ensemble review chose candidate %d of %d: %s",
        pick,
        len(groups),
        (getattr(verdict, "thought", "") or "")[:200],
    )
    return groups[pick]


def _record(
    question: str,
    candidates: list[_Candidate],
    groups: list[_Group],
    chosen: _Candidate,
    how: str,
) -> None:
    """Append every candidate and the choice to the candidates file.

    Written per question as one JSON line so a killed run keeps what it had,
    and keyed by the question text because the question *id* belongs to the
    harness and is not visible from in here.

    Rows are deliberately not stored -- a result set can be 10,000 rows and
    there are ten of them per question. ``group`` is the identity that matters:
    two candidates sharing a group returned exactly the same result, so the
    file can be re-voted and re-scored offline without re-executing anything.
    """
    path_text = config.ensemble_candidates_path()
    if not path_text:
        return
    group_of: dict[int, int] = {}
    for index, group in enumerate(groups):
        for member in group.members:
            group_of[id(member)] = index
    record = {
        "question": question,
        "decided_by": how,
        "chosen": chosen.spec.name,
        "n_results": len(groups),
        "votes": [group.votes for group in groups],
        "candidates": [
            {
                "generator": candidate.spec.name,
                "phase": candidate.phase,
                "sql": candidate.sql,
                "rows": candidate.row_count,
                "cols": list(candidate.columns),
                "truncated": candidate.truncated,
                "error": candidate.error,
                # -1 means the candidate never reached a group: it failed, or
                # it was empty while other candidates were not.
                "group": group_of.get(id(candidate), -1),
                "chosen": candidate is chosen,
            }
            for candidate in candidates
        ],
    }
    line = json.dumps(record, ensure_ascii=False)
    try:
        with _RECORD_LOCK:
            target = Path(path_text)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    except OSError:
        # Losing the record is not losing the answer.
        logger.warning("Could not append to the candidates file", exc_info=True)


def run_ensemble_sql_agent(
    llm: Any,
    messages: list[BaseMessage],
    toolbox: SQLAgentToolbox,
    state: Any,
    question: str = "",
) -> SQLAgentOutcome:
    """Explore once, write the query many ways from that, and answer by vote.

    Phase one is the deep agent, alone, exactly as the deep-agent arm runs it.
    Its query becomes the first candidate and its tool calls become the
    evidence. Phase two hands that evidence to the writers -- three methods,
    several each -- as single calls, and they are what the vote is between.

    Phase two is skipped when the explorer produced nothing to build on, since
    writers with no evidence are the single-call path with extra steps, and
    that path is measured below the explorer's own score.
    """
    deadline = time.monotonic() + config.ensemble_deadline_s()
    threshold = config.ensemble_review_threshold()
    trajectory: list[dict[str, Any]] = []

    # A separate executor from the toolbox's: candidates are whole answers
    # rather than probes, so they need a row cap high enough that two different
    # results are actually seen to differ, and a budget the whole pool cannot
    # exhaust halfway through.
    cap = config.ensemble_max_rows()
    runner = ProbeExecutor(
        connector=getattr(toolbox.executor, "connector", None),
        max_calls=config.ensemble_max_executions(),
        max_rows=cap,
        timeout_s=config.ensemble_timeout_s(),
    )

    candidates: list[_Candidate] = []
    try:
        independent = config.ensemble_independent_generators()
        if independent:
            # No shared evidence on this path, by design: the whole point is
            # that the generators do not read each other's conclusions. They
            # go straight to the vote, which is the same vote as ever.
            generated = _explore_independently(llm, messages, state, toolbox)
            candidates.extend(generated)
            evidence = ""
            trajectory.append(
                {
                    "phase": 1,
                    "mode": "independent",
                    "generators": len(generated),
                    "tool_calls": len(toolbox.calls),
                    "distinct_sql": len(
                        {" ".join(c.sql.split()).lower() for c in generated}
                    ),
                }
            )
        else:
            explorer = _explore(llm, messages, toolbox, state)
            if explorer is not None:
                candidates.append(explorer)
            evidence = _evidence_block(toolbox, config.ensemble_evidence_chars())
            trajectory.append(
                {
                    "phase": 1,
                    "explorer_sql": bool(explorer),
                    "tool_calls": len(toolbox.calls),
                    "evidence_chars": len(evidence),
                }
            )

        if evidence and not independent and time.monotonic() < deadline:
            specs = [
                _Spec(WRITER_ENGINE, method, order, replica=copy > 0, copy=copy)
                for order, method in enumerate(METHODS, start=1)
                for copy in range(config.ensemble_writers_per_method())
            ]
            writers = _write(llm, messages, specs, evidence)
            candidates.extend(writers)
            # ``distinct_sql`` is the point of the first run. Nine writers in
            # threes, none of them sampling, so this is three unless the
            # provider varies on its own -- and if it is three, the six
            # duplicates were bought and not received.
            trajectory.append(
                {
                    "phase": 2,
                    "writers": len(writers),
                    "distinct_sql": len({" ".join(w.sql.split()).lower() for w in writers}),
                    "by_method": {
                        method: len(
                            {
                                " ".join(w.sql.split()).lower()
                                for w in writers
                                if w.spec.method == method
                            }
                        )
                        for method in METHODS
                    },
                }
            )

        _execute(candidates, runner, cap)
        groups = _cluster(candidates)
        trajectory.append(
            {
                "candidates": len(candidates),
                "ran": sum(1 for c in candidates if c.ran),
                "results": len(groups),
                "votes": [g.votes for g in groups],
            }
        )

        n_results = len(groups)
        if not groups:
            # Nothing ran. Fall back to any SQL at all rather than answering
            # nothing: an unparseable query and no query score the same, but a
            # query can still be repaired by the loop downstream. The explorer
            # is first in the list, so this prefers its query.
            fallback = next((c for c in candidates if c.sql), None)
            if fallback is None:
                return SQLAgentOutcome(
                    result=None,
                    stop_reason="ensemble_no_candidates",
                    turns=len(candidates),
                    unconstructable_reason="No generator produced a query.",
                    trajectory=trajectory,
                )
            chosen, how = fallback, "no_result_fallback"
        elif _is_decided(groups, threshold):
            chosen, how = groups[0].best, "vote"
        else:
            reviewed = _review(llm, groups, question)
            chosen = (reviewed or groups[0]).best
            how = "review" if reviewed else "review_fallback"
    finally:
        runner.close()

    trajectory.append(
        {
            "decided_by": how,
            "generator": chosen.spec.name,
            "phase": chosen.phase,
            "candidates": len(candidates),
        }
    )
    _record(question, candidates, groups, chosen, how)
    logger.info(
        "Ensemble finished: %d candidates, %d distinct results, decided by %s "
        "(generator=%s, phase=%d)",
        len(candidates),
        n_results,
        how,
        chosen.spec.name,
        chosen.phase,
    )

    # ``response`` may not be empty, and the summary that belongs to the chosen
    # query is its own generator's. The thought is the next best thing; the
    # last resort only has to be real prose, which a rejected empty string and
    # a rejected ellipsis both are not.
    summary = (
        chosen.response
        or chosen.thought
        or (
            f"This query was selected by {how} from {len(candidates)} candidate "
            "queries that were each executed against the database."
        )
    )
    return SQLAgentOutcome(
        result=SQLGenerationModel(
            thought=chosen.thought or f"Chosen by {how} from {len(candidates)} candidates.",
            sql_code=chosen.sql,
            response=summary,
        ),
        stop_reason=f"ensemble_{how}",
        turns=len(candidates),
        trajectory=trajectory,
    )


__all__ = ["run_ensemble_sql_agent", "METHODS", "DEEP_ENGINE", "WRITER_ENGINE"]
