# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Construction as a deep agent that weighs competing readings of a question.

Why this exists rather than more turns for the tool-calling loop: measured on
866 paired questions the loop fixed 32 and broke 33, and the best query in its
own trajectory was the one it already submitted. Depth on a single reading is
spent. The candidate-pool run points the other way — seven drafts per question
held a correct query for 40% of the questions a single call gets wrong, worth
about +71 over 691 — but picking among them by result agreement realised -3,
because these errors are systematic and the wrong drafts agree with each other.

So the bet here is the one neither run made: enumerate the readings a question
genuinely admits, probe the database for the evidence that separates them, and
decide by argument rather than by vote. The sub-agent exists to make that
decision on evidence instead of on the main agent's own momentum.

Caps live in :mod:`.config`. The agent is off unless
``SQL_DEEP_AGENT_ENABLED`` says otherwise.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Sequence
from typing import Any, Callable, Optional

from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from auto_ontology.retrieval.text_to_sql.models import SQLGenerationModel
from auto_ontology.retrieval.text_to_sql.sql_agent import config, reflector
from auto_ontology.retrieval.text_to_sql.sql_agent.loop import (
    SQLAgentOutcome,
    _gate_submission,
    _truncate,
)
from auto_ontology.retrieval.text_to_sql.sql_agent.tools import SQLAgentToolbox
from auto_ontology.utils.llm_invoke import (
    RETRY_MAX_ATTEMPTS,
    RETRY_MAX_BACKOFF_S,
    _is_retryable,
)

logger = logging.getLogger(__name__)


def _invoke_with_retry(
    agent: Any, conversation: list[BaseMessage], deadline: float
) -> dict:
    """Run the agent, retrying the transient server errors.

    The agent's LLM calls are made by LangGraph, so none of them pass through
    :mod:`~auto_ontology.utils.llm_invoke` and none of them inherit its
    backoff. A single 429 therefore used to surface here as a bare exception
    and end the question at turn zero with no SQL at all, which scores exactly
    like the model being wrong: 82 of one BIRD run's 102 unanswered questions
    died this way, turning a +35 into a -34.

    Only retryable conditions are retried. A ``GraphRecursionError`` means the
    agent spent its step budget and must still reach the caller untouched.
    """
    last: Exception | None = None
    for attempt in range(RETRY_MAX_ATTEMPTS):
        try:
            return agent.invoke(
                {"messages": conversation},
                {"recursion_limit": config.deep_recursion_limit()},
            )
        except Exception as exc:  # noqa: BLE001
            if not _is_retryable(exc):
                raise
            last = exc
            wait = min(2 ** (attempt + 1) + random.uniform(0, 1), RETRY_MAX_BACKOFF_S)
            # Waiting past the deadline only to be cut off on return wastes
            # the budget the remaining questions could have used.
            if attempt == RETRY_MAX_ATTEMPTS - 1 or time.monotonic() + wait > deadline:
                break
            logger.warning(
                "Deep agent hit a retryable error on attempt %d/%d — "
                "retrying in %.1fs: %s",
                attempt + 1,
                RETRY_MAX_ATTEMPTS,
                wait,
                exc,
            )
            time.sleep(wait)
    raise last if last else RuntimeError("deep agent invocation made no attempt")


#: Handed to ``create_deep_agent``, which places its system prompt ahead of the
#: conversation. Deliberately thin: the construction prompt the single-call
#: path builds is itself a stack of system messages, the last of which tells
#: the model to produce the query, so anything substantive placed *before* them
#: is read first and overridden last. :data:`PROTOCOL_PROMPT` therefore goes in
#: at the end of the conversation instead — the same placement, and for the
#: same reason, as the tool-calling loop's own protocol.
ROLE_PROMPT = """\
You are writing one SQL query to answer a database question. The schema, the
dialect rules and the question itself follow. The protocol you must work by is
stated in the last system message.
"""


#: Opens the protocol and is never reordered. The first two paragraphs frame
#: what the whole protocol is for, and the minimality paragraph governs every
#: block after it rather than sitting alongside them, so moving it would change
#: what the prompt says and not merely the order it says it in.
_PROTOCOL_HEAD = """\
You write one SQL query answering the user's question, for the schema and
dialect given above.

Most questions that go wrong here are not wrong about SQL. They are wrong
about which of several defensible readings the question meant — which column
counts as "the name", whether a filter literal matches what is stored, whether
one row means one entity or one pairing.

Before anything else, understand what is being asked of you, because it is
narrower than it looks. You are not asked for the most useful answer, or the
most careful one, or the one a good analyst would hand a colleague. You are
asked for the answer to the question as written, and the correct query is
almost always the smallest one that produces it: the fewest output columns,
the fewest conditions, the least cleverness. Every element you add that the
question did not ask for is a way to be wrong. If you find yourself adding a
guard because a row looks like it does not belong, excluding nulls the
question never mentioned, returning a column because it makes the output
easier to read, or deduplicating because duplicates look untidy, stop: none
of those were asked for, and each one changes the answer. Being thorough is
how this task is lost.

With that in mind, work in this order."""


#: Closes the protocol and is never reordered: it tells the model to answer,
#: so anything placed after it is read after the instruction to stop.
_PROTOCOL_TAIL = """\
When you are done, return the final query with your reasoning. Your answer
must be a single statement, read-only, and executable as written."""


#: The orderable body of the protocol, one entry per self-contained block.
#:
#: Two pairs of paragraphs are deliberately held together as single blocks
#: rather than split further. ``readback`` keeps the FROM paragraph with the
#: SELECT one because it opens "Then read the FROM clause back the same way",
#: and ``as_stored`` keeps the percentage rule with the casting rule because it
#: opens "When that formula is a percentage". Separating either would mean
#: rewriting the text, which would confound a change of wording with a change
#: of order -- the one thing this decomposition exists to keep apart.
_PROTOCOL_BLOCKS: dict[str, str] = {
    "readings": """\
state the readings. Before writing any query, name the points where the
question is genuinely open, and for each the readings a careful person could
take. Be concrete: "'largest' could mean population or area", not "the wording
is ambiguous". If a question is truly unambiguous, say so and move on — do not
manufacture doubt.""",
    "evidence": """\
settle them with evidence, not preference. For each open point, decide
what in the database would tell you the answer, and go and look with the tools.
Whether a status column stores 'OWNER' or 'Owner' is a fact you can check.
Which column a question means by "name" is often settled by what the other
candidate actually contains. Prefer the reading the evidence supports over the
reading you thought of first.""",
    "only_asked": """\
Answer only what was asked. Do not add a condition the question does not
state, even when adding it would make the answer more defensible. If a table
mixes school rows with district totals and the question says "school", you may
well be right that the totals do not belong — but the question did not ask you
to exclude them, and a query that answers a better question is still the wrong
answer. Restrict rows only where the question, or the evidence, says to.""",
    "formula": """\
Where the evidence gives a formula, write that formula. Do not swap it for a
stored column that holds the same quantity, even after confirming the two
agree: a ratio computed from its parts and a ratio read from a column are
compared digit by digit, and they do not match. Compute what you were told to
compute.""",
    "two_readings": """\
when two readings survive the evidence, write the query for each and
run both. Then take the one that assumes less — fewer conditions, fewer
columns, less interpretation — not the one whose result looks more plausible.
Do not decide by how many of your drafts agree either: your drafts share your
assumptions, so agreement among them is not evidence.""",
    "grain": """\
decide the grain deliberately, in whichever direction the question
points. If the query joins, ask what a single output row stands for and
whether a join can emit the same entity more than once. A question counting
entities needs a DISTINCT aggregate when a join can repeat them; a question
listing values needs DISTINCT when it asks which values occur rather than for
every row. But a question that asks for rows wants rows, and deduplicating
them is as wrong as failing to. Decide from the question, not from how the
result looks.""",
    "readback": """\
before you answer, read your SELECT list back against the question.
Count the things the question asks to be returned, then count the columns you
are returning. Where the two numbers differ it is almost always the query that
is wrong, and almost always too wide. Where they agree, check the order is the
order the question named them in, because two columns in the wrong order is a
wrong answer. "List the transactions" wants what identifies a transaction, not
every column of the table. "Give the address" wants the column called the
address, not the four columns that together make a postal address. If you are
returning something the question did not name, remove it.

Then read the FROM clause back the same way, asking of each column where you
are reading it from. When the column you want exists in more than one of the
tables in front of you, those copies are not interchangeable — the same name
holds different values in different tables, and reading it from the table
you have already joined instead of the one the question is about is a wrong
answer that runs without error. Decide which table the question is talking
about, and read the column from that one.""",
    "as_stored": """\
Return each value as it is stored. A `CAST` or a rounding applied on the way
out changes how the number reads and so changes the answer, and no question
asks for that. The exception is arithmetic that needs it to be correct at all
— an average or a share computed from integers — and there the evidence block
states the formula, cast included, so write what it states and add nothing
to it.

When that formula is a percentage, scale before dividing: write
`CAST(<numerator> AS REAL) * 100 / <denominator>`, never
`CAST(<numerator> AS REAL) / <denominator> * 100`. The two are the same
number in arithmetic and different numbers in binary floating point, apart
in the last digit, and the answer is compared exactly.""",
    "evidence_auth": """\
Where the evidence block defines a term, a formula or a literal, use that
definition rather than the one you would have chosen: the evidence is
authoritative and your reading of the schema is not.""",
    "analyses": """\
Note what the supplied analyses are for. When one of them says a complete
address is Street, City, State and Zip, or that a full name is two columns,
it is telling you how the data is laid out — not instructing you to return
all of those. A question asking for the street address wants the street
column. Use the analyses to find the right column; let the question decide
how many.""",
    "tools": """\
Use `repair_sql` when a draft errors and the cause is not obvious — it
classifies the failure and tells you what to look for. Use `run_sql` freely;
reading data is cheap and guessing is not. A query that returns no rows is
almost always a filter literal that does not match what is stored, so find
which one by removing filters until rows appear.""",
}


#: Blocks written as a numbered step. The number is not stored with the text:
#: it is assigned from the block's position among the other numbered blocks, so
#: a reordering renumbers rather than producing a prompt whose "Fifth" arrives
#: before its "Second".
_ORDINAL_BLOCKS = ("readings", "evidence", "two_readings", "grain", "readback")

_ORDINALS = ("First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh")

#: The order the blocks are emitted in when nothing overrides it. Reproduces
#: the prompt exactly as it read before it was decomposed, which
#: ``tests/test_protocol_order.py`` asserts.
PROTOCOL_DEFAULT_ORDER = tuple(_PROTOCOL_BLOCKS)


def protocol_prompt(order: Optional[Sequence[str]] = None) -> str:
    """Assemble the protocol, optionally with the body blocks reordered.

    ``order`` names the body blocks, in the order they should appear. Names
    that are not blocks are ignored and blocks the order leaves out are
    appended in their default position, so a partial or stale order degrades
    to something close to the default rather than to a prompt with a rule
    silently missing from it.

    Falls back to :func:`config.deep_protocol_order` when ``order`` is None,
    which is how an experiment varies the order across runs of one build.
    """
    if order is None:
        order = config.deep_protocol_order()
    names = [name for name in (order or ()) if name in _PROTOCOL_BLOCKS]
    seen = set(names)
    names.extend(name for name in PROTOCOL_DEFAULT_ORDER if name not in seen)

    ordinal_rank = {
        name: index
        for index, name in enumerate(n for n in names if n in _ORDINAL_BLOCKS)
    }
    parts = [_PROTOCOL_HEAD]
    for name in names:
        body = _PROTOCOL_BLOCKS[name]
        if name in ordinal_rank:
            rank = ordinal_rank[name]
            word = _ORDINALS[rank] if rank < len(_ORDINALS) else f"Step {rank + 1}"
            body = f"{word}, {body}"
        parts.append(body)
    parts.append(_PROTOCOL_TAIL)
    return "\n\n".join(parts) + "\n"


#: The default-ordered protocol, kept as a module constant for the callers and
#: tests that read it. The agent itself calls :func:`protocol_prompt` so that
#: the order can be set per run without rebuilding.
PROTOCOL_PROMPT = protocol_prompt(PROTOCOL_DEFAULT_ORDER)


ADJUDICATOR_PROMPT = """\
You decide which of two candidate SQL queries answers the question that was
actually asked. You are given the question, both queries and both result sets.

You are here because the agent that wrote them cannot judge them neutrally: it
produced both from the same assumptions, so whichever it leaned towards it will
keep leaning towards. Your job is to read the question as written and say which
query matches it.

Work from the question's words. If it asks "how many players", a result that
counts appearances rather than players is wrong however reasonable its SQL
looks. If it names a unit, a currency or a year, check the query honours it. If
it asks for one thing and a query returns several columns, that matters.

Judge the queries, not the results' appearance. A short result is not more
likely to be right than a long one, a round number is not evidence, and
neither is a result that merely looks like what an answer should look like.

Where the two differ by a condition only one of them applies, the one without
it is usually right. The agent that wrote them tends to add guards to be
careful — excluding rows it judges not to belong, skipping nulls, filtering a
type column the question never mentioned — and a question that did not ask for
that exclusion is not answered better by it. Endorse the extra condition only
if the question or the evidence actually calls for it.

If one is clearly right, say which and why, in one or two sentences. If the
question genuinely cannot distinguish them, say which is the safer reading and
why. Never answer with both.
"""


class _SQLArg(BaseModel):
    sql: str = Field(description="A single read-only SQL statement.")


class _SQLPurposeArg(BaseModel):
    sql: str = Field(description="A single read-only SQL statement.")
    purpose: str = Field(
        default="", description="What you expect to learn by running it."
    )


class _TableArg(BaseModel):
    table_name: str = Field(description="Table to describe, as named in the schema.")


class _SearchArg(BaseModel):
    query: str = Field(
        description="What you are looking for, in words — e.g. 'player salaries'."
    )


class _RepairArg(BaseModel):
    sql: str = Field(description="The draft that failed.")
    error: str = Field(description="The error it produced, verbatim.")


def _repair_tool(state: Any) -> Callable[[str, str], str]:
    """Reconstruction as a tool, rather than only as a downstream repair node.

    The graph already has a reconstruction agent, but it runs *after* a query
    has failed validation and the construction turn is over. Its error
    classifier is useful mid-draft too, so it is offered here on the same
    terms as the probes: the agent may call it, and it only explains — the
    rewriting stays with the agent.
    """

    def repair_sql(sql: str, error: str) -> str:
        # Imported here because the reconstruction module imports the graph's
        # agent base, and importing it at module scope makes the sql_agent
        # package depend on the node it is a sibling of.
        from auto_ontology.retrieval.text_to_sql.agents.sql_reconstruction import (
            SQLReconstructionAgent,
        )

        path_state = state.get("path_state") or {}
        try:
            analysis = SQLReconstructionAgent()._analyze_error(
                state,
                question=path_state.get("question") or "",
                error_context=f"Query:\n{sql}\n\nError:\n{error}",
                existing_tables=path_state.get("relevant_tables") or [],
            )
        except Exception as exc:  # noqa: BLE001 — a tool must not kill the turn
            logger.warning("repair_sql failed: %s", exc)
            return f"The repair analyser failed ({exc}). Diagnose it yourself."

        lines = [f"Error type: {analysis.error_type.value}", analysis.explanation]
        for query in getattr(analysis, "search_queries", None) or []:
            lines.append(f"Consider searching for: {query}")
        return "\n".join(line for line in lines if line)

    return repair_sql


def build_tools(toolbox: SQLAgentToolbox, state: Any) -> list[StructuredTool]:
    """The toolbox as LangChain tools, plus reconstruction.

    ``submit_sql`` is deliberately absent: the deep agent terminates by
    returning its structured answer, and the gate that ``submit_sql`` enforced
    is applied to that answer in :func:`run_deep_sql_agent` instead. Keeping it
    as a tool as well would give the agent two ways to finish and a reason to
    take neither.
    """
    return [
        StructuredTool.from_function(
            func=toolbox.run_sql,
            name="run_sql",
            description=(
                "Execute a read-only query and return its rows. Use it to check "
                "what is actually stored before committing to a filter or column."
            ),
            args_schema=_SQLPurposeArg,
        ),
        StructuredTool.from_function(
            func=toolbox.check_sql,
            name="check_sql",
            description=(
                "Statically check a draft against the schema without running it: "
                "unknown columns, unjoined tables, write statements."
            ),
            args_schema=_SQLArg,
        ),
        StructuredTool.from_function(
            func=toolbox.diagnose_sql,
            name="diagnose_sql",
            description=(
                "Probe a draft against live data and report what looks wrong — "
                "filters matching nothing, joins that drop every row."
            ),
            args_schema=_SQLArg,
        ),
        StructuredTool.from_function(
            func=toolbox.inspect_table,
            name="inspect_table",
            description=(
                "Show a table's columns, types and sample values. Use it when a "
                "column's meaning or stored format matters."
            ),
            args_schema=_TableArg,
        ),
        StructuredTool.from_function(
            func=toolbox.search_tables,
            name="search_tables",
            description=(
                "Find tables beyond the ones supplied, by description. Use it "
                "when the schema given has nothing holding what you need."
            ),
            args_schema=_SearchArg,
        ),
        StructuredTool.from_function(
            func=_repair_tool(state),
            name="repair_sql",
            description=(
                "Classify why a draft failed and what to look for. Use it when a "
                "draft errors and the cause is not obvious from the message."
            ),
            args_schema=_RepairArg,
        ),
    ]


#: Tools the deep-agent defaults bind that this task has no use for. The
#: virtual filesystem is there for work that outgrows a context window;
#: construction is one question against one schema and the loop it replaces
#: finished in a median of four turns, so nothing here would ever be written
#: down and read back. They are dropped because tool choice is the scarce
#: resource in this agent — the whole bet is that it spends its attention
#: comparing readings of the question — and eight unusable options are eight
#: chances to spend a turn elsewhere. ``execute`` is inert on the default
#: state backend regardless, which has no such method.
_UNUSED_DEFAULT_TOOLS = frozenset(
    {"ls", "read_file", "write_file", "edit_file", "glob", "grep", "delete", "execute"}
)


def _strip_default_tools(llm: BaseChatModel) -> None:
    """Register a harness profile dropping the tools in
    :data:`_UNUSED_DEFAULT_TOOLS`.

    Profiles are resolved per model provider, so the key has to be derived
    from the model we were handed. Best-effort on purpose: if the provider
    cannot be resolved the agent still works, it just carries the extra tools,
    and that is not worth failing a question over.
    """
    try:
        from deepagents import HarnessProfileConfig, register_harness_profile
        from deepagents._models import get_model_identifier, get_model_provider

        provider = get_model_provider(llm)
        identifier = get_model_identifier(llm)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not resolve a harness profile key: %s", exc)
        return

    config_kwargs: dict[str, Any] = {"excluded_tools": _UNUSED_DEFAULT_TOOLS}
    try:
        from deepagents import GeneralPurposeSubagentProfile

        # The default general-purpose sub-agent would arrive holding the same
        # filesystem tools, so dropping them on the main agent alone would
        # only move them one call away.
        config_kwargs["general_purpose_subagent"] = GeneralPurposeSubagentProfile(
            enabled=False
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not disable the general-purpose subagent: %s", exc)

    profile = HarnessProfileConfig(**config_kwargs)
    for key in (
        f"{provider}:{identifier}" if provider and identifier else None,
        provider,
    ):
        if key:
            try:
                register_harness_profile(key, profile)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Could not register harness profile %r: %s", key, exc)


def _build_agent(llm: BaseChatModel, tools: list[StructuredTool]):
    """Compile the deep agent.

    ``ToolStrategy`` rather than the default auto-detection: the final answer
    comes back through a tool call, which is the one structured-output path the
    gateway in front of this model is known to handle. Provider-native
    structured output is what ``StrictLLMOutputModel`` exists to clean up after.
    """
    from deepagents import create_deep_agent

    _strip_default_tools(llm)
    subagents = []
    if config.is_deep_adjudicator_enabled():
        subagents.append(
            {
                "name": "adjudicate_readings",
                "description": (
                    "Decide which of two candidate queries answers the question "
                    "as asked. Give it the question, both queries and both "
                    "result sets. Use it when two readings survive the evidence."
                ),
                "system_prompt": ADJUDICATOR_PROMPT,
                "model": llm,
                # Isolated so the main agent's reasoning for its preferred
                # draft does not travel with the question it is asking about.
                # A critic handed the argument it is meant to weigh is not a
                # second opinion. Measured, this cut the wrong way: see
                # ``config.is_deep_adjudicator_enabled``.
                "mode": "isolated",
            }
        )
    return create_deep_agent(
        model=llm,
        tools=tools,
        system_prompt=ROLE_PROMPT,
        response_format=ToolStrategy(SQLGenerationModel),
        subagents=subagents or None,
    )


def _normalise(sql: str, toolbox: SQLAgentToolbox) -> str:
    """Quote the mixed-case identifiers the gate would otherwise bounce over.

    Measured on the first 38 questions of a full run, 28 of them — 74% — were
    sent back for this and this alone, which doubled their cost and spent the
    bounce budget that exists for semantic repair on quote marks. The gate was
    never really asking a question here: it computes the corrected form itself
    and every other check it runs already reads that form rather than what the
    model typed. So the rewrite is applied instead of requested.

    Deterministic and schema-driven, so it cannot invent an identifier: it
    quotes names it found in the catalog and leaves everything else alone.
    """
    try:
        from auto_ontology.retrieval.text_to_sql.agents.sql_parse_validation import (
            quote_known_mixed_case_identifiers,
        )

        return quote_known_mixed_case_identifiers(
            sql, toolbox.relevant_tables, toolbox.dialect
        )
    except Exception as exc:  # noqa: BLE001 — a cosmetic fix must not cost a question
        logger.debug("Could not normalise identifier quoting: %s", exc)
        return sql


def _steps(messages: list[Any]) -> list[dict[str, Any]]:
    """The agent's reasoning and tool use, flattened for offline reading.

    Counts of messages say whether the agent ran; they do not say whether it
    thought correctly, which is the only question worth asking of a run that
    comes out level. Each assistant turn keeps its prose and the calls it made,
    each tool result keeps its head — enough to replay a wrong answer and see
    which step went wrong, and bounded so 1,534 of these stay readable.
    """
    steps: list[dict[str, Any]] = []
    for message in messages or []:
        calls = getattr(message, "tool_calls", None) or []
        text = getattr(message, "content", "") or ""
        if isinstance(text, list):
            # Content blocks rather than a plain string, which is what the
            # gateway returns when the model interleaves prose and tool use.
            text = " ".join(
                part.get("text", "") for part in text if isinstance(part, dict)
            )
        name = getattr(message, "name", None)
        kind = message.__class__.__name__
        if kind == "ToolMessage":
            steps.append(
                {"step": "tool_result", "name": name, "result": _truncate(str(text))}
            )
        elif kind != "AIMessage":
            # The construction prompt arrives as a stack of system messages and
            # the question as a human one. They are the same for every question
            # and are not the agent's reasoning, so logging them would bury the
            # three lines that are in several kilobytes that are not.
            continue
        elif calls or text.strip():
            steps.append(
                {
                    "step": "thought",
                    "text": _truncate(text.strip(), 2000),
                    "calls": [
                        {
                            "name": call.get("name"),
                            # The sub-agent's brief is the interesting argument
                            # here; a 2kB SQL draft is already in the result.
                            "args": _truncate(str(call.get("args") or {}), 600),
                        }
                        for call in calls
                    ],
                }
            )
    return steps


def _answer(state: dict) -> Optional[SQLGenerationModel]:
    """Pull the structured answer out of whatever the agent returned."""
    candidate = (state or {}).get("structured_response")
    if isinstance(candidate, SQLGenerationModel):
        return candidate
    if isinstance(candidate, dict):
        try:
            return SQLGenerationModel(**candidate)
        except Exception:  # noqa: BLE001
            return None
    return None


def _evidence(state: Any) -> str:
    """The question's evidence block, from wherever this graph keeps it.

    Checked in both places because the field moved: it used to be embedded in
    the question text and now arrives as its own state field, with
    ``path_state`` still carrying a copy on some paths.
    """
    if not isinstance(state, dict):
        return ""
    direct = state.get("evidence")
    if isinstance(direct, str) and direct.strip():
        return direct
    nested = (state.get("path_state") or {}).get("evidence")
    return nested if isinstance(nested, str) else ""


def run_deep_sql_agent(
    llm: BaseChatModel,
    messages: list[BaseMessage],
    toolbox: SQLAgentToolbox,
    state: Any,
) -> SQLAgentOutcome:
    """Construct one SQL candidate with the deep agent, then gate it.

    The gate is the loop's own, applied here to the structured answer rather
    than to a ``submit_sql`` call. It earned its place: over 866 questions it
    bounced a quarter of the submissions it later passed, and the checks it
    runs are the ones that catch scoring errors the model declines to look for
    itself. Findings are recorded for observability, but do not trigger another
    generation: the deep-agent path produces exactly one final SQL candidate.
    """
    agent = _build_agent(llm, build_tools(toolbox, state))

    deadline = time.monotonic() + config.deadline_s()
    # Protocol last, after the construction prompt's own system messages and
    # the question. Those end by asking for the query, so a protocol placed
    # before them is the instruction the model reads first and abandons last.
    conversation: list[BaseMessage] = list(messages) + [
        SystemMessage(content=protocol_prompt())
    ]
    trajectory: list[dict[str, Any]] = []
    answer: Optional[SQLGenerationModel] = None
    stop_reason = "answered"

    for attempt in range(1):
        if time.monotonic() > deadline:
            stop_reason = "deadline"
            break
        try:
            result = _invoke_with_retry(agent, conversation, deadline)
        except Exception as exc:  # noqa: BLE001
            # Includes GraphRecursionError, which means the agent spent its
            # step budget without answering. Treated as a stop rather than an
            # error: the caller falls back to the single call.
            logger.warning("Deep agent invocation failed: %s", exc)
            stop_reason = "error"
            break

        steps = _steps(result.get("messages") or [])
        trajectory.append(
            {
                "attempt": attempt,
                "messages": len(result.get("messages") or []),
                "todos": result.get("todos"),
                # How often the adjudicator is actually consulted decides
                # whether this agent is the bet it was built as or the loop
                # with a longer prompt, so it is counted rather than inferred.
                "subagent_calls": sum(
                    1
                    for step in steps
                    for call in step.get("calls") or []
                    if call.get("name") == "task"
                ),
                "steps": steps,
            }
        )
        answer = _answer(result)
        if answer is not None:
            answer.sql_code = _normalise(answer.sql_code, toolbox)
            trajectory[-1]["answer_thought"] = _truncate(answer.thought, 2000)
            trajectory[-1]["answer_sql"] = _truncate(answer.sql_code, 1200)
        if answer is None:
            stop_reason = "no_answer"
            break

        findings = _gate_submission(answer.sql_code, toolbox, first_pass=(attempt == 0))
        if not findings:
            break

        trajectory[-1]["gate"] = findings
        stop_reason = "gate_exhausted"
        break

    # After the gate, not before it: the gate's findings are the author's to
    # act on with its remaining turns, and a query still being revised is not
    # the one worth a second reader's time. A review here sees what would
    # otherwise have been returned.
    if answer is not None and config.reflector_enabled():
        edited, record = reflector.reflect(
            llm,
            question=getattr(toolbox, "question", "") or "",
            evidence=_evidence(state),
            sql=answer.sql_code,
            toolbox=toolbox,
        )
        if trajectory:
            trajectory[-1]["reflector"] = record
        if record.get("applied"):
            answer.sql_code = edited

    return SQLAgentOutcome(
        result=answer,
        stop_reason=stop_reason,
        turns=len(trajectory),
        unconstructable_reason=(
            "" if answer else "The deep agent did not return a query."
        ),
        trajectory=trajectory,
    )


__all__ = [
    "ADJUDICATOR_PROMPT",
    "PROTOCOL_DEFAULT_ORDER",
    "PROTOCOL_PROMPT",
    "ROLE_PROMPT",
    "build_tools",
    "protocol_prompt",
    "run_deep_sql_agent",
]
