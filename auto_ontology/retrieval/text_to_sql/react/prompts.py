# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ReAct agent's system prompt and tool definitions.

Short on purpose. Rules about how SQL should look belong in code — the checks
``run_sql`` and ``submit`` apply — not here: rule blocks in the writer prompt
lost accuracy every time they were measured, and a policy being trained should
learn conventions from reward, not from a list.
"""

from __future__ import annotations

from auto_ontology.retrieval.text_to_sql.react.mode import EpisodeMode, ToolName
from auto_ontology.retrieval.text_to_sql.react.retrieve import KIND_LABELS

SYSTEM_PROMPT = """\
You answer a question about a database by writing one {dialect} SQL query.

You start with no schema. Find what you need with the tools:
- retrieve: search the ontology. Write your own search text: the question, \
something the last result was missing ("how is revenue calculated", "which \
column stores admission status"), or an exact table or column name. Search \
again whenever something is missing.
- run_sql: check a query against the catalog and run it. Use it to look at \
stored values before filtering on them, and to test a draft.
- submit: give your final query. It is checked and run; if it fails you get \
the error and can keep working.

Return exactly the requested output fields and no others.

Use only tables and columns a retrieve result showed you. A SIMILAR line names \
tables that hold nearly the same data; compare them with run_sql before you \
choose one."""

PREDICT_SYSTEM_PROMPT = """\
You answer a question that asks what will happen in a database's data, with a \
KumoRFM prediction.

You start with no schema. Find what you need with the tools:
- retrieve: search the ontology. Find the entity the question is about \
(customers, users, products) and the tables that record its history \
(orders, events, transactions). Search again whenever something is missing.
- predict: build a graph over every table retrieve has shown you and answer \
the question. If it is refused, retrieve what it says is missing and call it \
again.

Call predict once the entity and its history tables are in view."""

_PREDICT_WHEN = """

If the question asks what will happen (a future value, a likelihood, churn, \
the next purchase), do not write SQL: retrieve the entity and the tables that \
record its history, then call predict."""

USER_TEMPLATE = """\
QUESTION
{question}
{evidence_block}"""

NUDGE = "Call a tool: retrieve, run_sql, or submit your final query."

LAST_CALL = "No tool calls left. Call submit now with your best query."

PREDICT_LAST_CALL = "No tool calls left. Call predict now."

_KINDS = sorted(KIND_LABELS)

RETRIEVE_TOOL = {
    "type": "function",
    "function": {
        "name": ToolName.RETRIEVE,
        "description": (
            "Search tables, columns, definitions and analyses by meaning "
            "and by name. Returns matching tables with their columns, stored "
            "values and references, definitions with their SQL, how new "
            "tables join to ones already found, and tables that are near "
            "copies of them."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What you are looking for, in your own words or as an exact name.",
                },
                "kinds": {
                    "type": "array",
                    "items": {"type": "string", "enum": _KINDS},
                    "description": "Optional: search only these kinds.",
                },
            },
            "required": ["query"],
        },
    },
}

RUN_SQL_TOOL = {
    "type": "function",
    "function": {
        "name": ToolName.RUN_SQL,
        "description": (
            "Check a read-only query for unknown tables and columns, then run it "
            "and return the first rows."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "One SELECT query."}
            },
            "required": ["sql"],
        },
    },
}

SUBMIT_TOOL = {
    "type": "function",
    "function": {
        "name": ToolName.SUBMIT,
        "description": "Submit the final query that answers the question.",
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "The final SELECT query."},
                "response": {
                    "type": "string",
                    "description": (
                        "Two to four plain-English sentences for the user: what is "
                        "calculated, from which data, with which filters."
                    ),
                },
                "assumptions": {
                    "type": "string",
                    "description": "Any choice the question left open, and why.",
                },
            },
            "required": ["sql", "response"],
        },
    },
}

PREDICT_TOOL = {
    "type": "function",
    "function": {
        "name": ToolName.PREDICT,
        "description": (
            "Answer a question about the future with a KumoRFM prediction over "
            "the tables retrieve has shown. Ends the episode once it runs."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

TOOLS = [RETRIEVE_TOOL, RUN_SQL_TOOL, SUBMIT_TOOL]


def tools_for(mode: EpisodeMode, *, last_turn: bool = False) -> list[dict]:
    """The tools offered in *mode*; on the last turn, only the ones that answer."""
    if mode == EpisodeMode.PREDICT:
        return [PREDICT_TOOL] if last_turn else [RETRIEVE_TOOL, PREDICT_TOOL]
    if mode == EpisodeMode.AUTO:
        return [SUBMIT_TOOL, PREDICT_TOOL] if last_turn else [*TOOLS, PREDICT_TOOL]
    return [SUBMIT_TOOL] if last_turn else TOOLS


def system_prompt_for(mode: EpisodeMode, dialect: str | None) -> str:
    if mode == EpisodeMode.PREDICT:
        return PREDICT_SYSTEM_PROMPT
    prompt = SYSTEM_PROMPT.format(dialect=dialect or "SQL")
    return prompt + _PREDICT_WHEN if mode == EpisodeMode.AUTO else prompt


def last_call_for(mode: EpisodeMode) -> str:
    return PREDICT_LAST_CALL if mode == EpisodeMode.PREDICT else LAST_CALL


def nudge_for(mode: EpisodeMode) -> str:
    if mode == EpisodeMode.PREDICT:
        return "Call a tool: retrieve, or predict once the tables are in view."
    if mode == EpisodeMode.AUTO:
        return "Call a tool: retrieve, run_sql, submit your final query, or predict."
    return NUDGE


def out_of_calls_for(mode: EpisodeMode) -> str:
    if mode == EpisodeMode.PREDICT:
        return "No tool calls left; predict."
    if mode == EpisodeMode.AUTO:
        return "No tool calls left; submit or predict."
    return "No tool calls left; submit."


def user_message(question: str, evidence: str | None) -> str:
    evidence = (evidence or "").strip()
    evidence_block = f"\nEVIDENCE\n{evidence}\n" if evidence else ""
    return USER_TEMPLATE.format(
        question=question.strip(), evidence_block=evidence_block
    )
