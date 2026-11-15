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

from auto_ontology.retrieval.text_to_sql.react import config
from auto_ontology.retrieval.text_to_sql.react.mode import (
    ANSWERING_TOOLS,
    EpisodeMode,
    ToolName,
)
from auto_ontology.retrieval.text_to_sql.react.information import PURPOSES
from auto_ontology.retrieval.text_to_sql.react.retrieve import KIND_LABELS

_SQL_OPENING = """\
You answer a question about a database by writing one {dialect} SQL query."""

_INFORMATION_OPENING = """\
You answer questions about this organization's data from the database or its \
catalog. By default, write one {dialect} SQL query that computes the answer.

An information question asks about the catalog itself: which tables or columns \
exist, what one means, how tables connect, or how a measure is defined. Answer it, \
a greeting, or a question about what you can do in words: look it up with retrieve \
and purpose=information (an empty query lists every table, or every term with \
kinds=["term"]), then submit with sql empty and the answer in response.

Anything that needs values stored in the data (a count, a total, a list of \
values, which rows) needs SQL, since retrieve shows only samples. If unsure, write \
SQL.

In an information answer, use the exact table, column and definition names retrieve \
showed, and say only what it showed. If retrieve did not find what was asked, say so."""

SYSTEM_PROMPT = """\
{opening}

You start with no schema. Find what you need with the tools:
- retrieve: search the semantic layer: the business concepts the tables store, \
what each column means, and named calculations. Write your own search text in \
business words: the question, something the last result was missing ("how is \
revenue calculated", "which column stores admission status"), or an exact \
table or column name. What it finds comes back as tables and columns with \
their descriptions; a column that "references" another table is a foreign \
key to join on. Search again \
whenever something is missing.
- run_sql: check a query against the catalog and run it. Use it to look at \
stored values before filtering on them, and to test a draft.
- submit: give your final query. It is checked and run; if it fails you get \
the error and can keep working.

Return exactly the requested output fields and no others.

Use only tables and columns a retrieve result showed you. A SIMILAR line names \
tables that hold nearly the same data; compare them with run_sql before you \
choose one."""

_GREP_LINE = """\
- grep: search all rows of a run_sql result too long to show (run_sql says \
which page it saved it as), or read rows past the first ones with offset.
"""

_SUBMIT_LINES = """\
- run_sql: check a query against the catalog and run it. Use it to look at \
stored values before filtering on them, and to test a draft.
- submit: give your final query. It is checked and run; if it fails you get \
the error and can keep working.
"""

_FINAL_RUN_LINES = """\
- run_sql: check a query against the catalog and run it. Use it to look at \
stored values before filtering on them, and to test a draft. When you have the \
query that answers the question, run it with final=true: if it runs cleanly it \
is your answer; if it fails you get the error and can keep working.
"""

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


def custom_prompts_section(
    custom_prompts: str | None, acronyms: list[dict[str, str]] | None
) -> str:
    """The organization's custom prompts and acronyms for the system prompt.

    "" when neither is set, so a deployment without them sees no new section.
    """
    parts = []
    if text := (custom_prompts or "").strip():
        parts.append(f"ORGANIZATION INSTRUCTIONS\n{text}")
    definitions = [
        f"- {name}: {(entry.get('description') or '').strip()}"
        for entry in acronyms or []
        if (name := (entry.get("name") or "").strip())
    ]
    if definitions:
        parts.append(
            "GLOSSARY\nDefinitions the user's organization has registered. Use "
            "them to resolve abbreviations, shortcuts, and internal jargon in the "
            "question.\n" + "\n".join(definitions)
        )
    return "\n\n".join(parts)


USER_TEMPLATE = """\
QUESTION
{question}
{evidence_block}"""

NUDGE = "Call a tool: retrieve, run_sql, or submit your final query."

INFORMATION_NUDGE = (
    "Call a tool: retrieve, run_sql, or submit your final query. If this is an "
    "information question, submit with sql empty and the answer in response."
)

LAST_CALL = "No tool calls left. Call submit now with your best query."

INFORMATION_LAST_CALL = (
    "No tool calls left. Call submit now with your best query, or with sql empty "
    "and the answer in response if this is an information question."
)

FINAL_RUN_NUDGE = (
    "Call a tool: retrieve, run_sql, or run_sql with final=true on your final query."
)

FINAL_RUN_LAST_CALL = (
    "No tool calls left. Call run_sql with final=true now with your best query."
)

PREDICT_LAST_CALL = "No tool calls left. Call predict now."

_KINDS = sorted(KIND_LABELS)

RETRIEVE_TOOL = {
    "type": "function",
    "function": {
        "name": ToolName.RETRIEVE,
        "description": (
            "Search the semantic layer by meaning and by name: terms (the "
            "business concepts a table stores, such as Customer or Order), "
            "attributes (what a column means, such as Order Date), "
            "definitions (named calculations with their SQL) and analyses. "
            "Tables and columns the semantic layer does not describe, such as "
            "foreign keys and link tables, are searched directly. An exact "
            "table or column name is matched too. Every term and "
            "attribute found comes back as the table that stores it, with its "
            "columns, stored values and references; definitions come with "
            "their SQL. Also returns how new tables join to ones already "
            "found, and tables that are near copies of them."
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
                    "description": (
                        "Optional: search only these kinds. term finds tables, "
                        "attribute finds columns, foreign keys included."
                    ),
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

GREP_TOOL = {
    "type": "function",
    "function": {
        "name": ToolName.GREP,
        "description": (
            "Search a saved run_sql result: the rows matching a pattern, each "
            "with its row number. With no pattern, reads rows in order."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "page": {
                    "type": "integer",
                    "description": "The page run_sql saved the result as.",
                },
                "pattern": {
                    "type": "string",
                    "description": "Text or a regular expression, case-insensitive. Empty for all rows.",
                },
                "offset": {
                    "type": "integer",
                    "description": "Skip this many matching rows first, to read further.",
                },
            },
            "required": ["page"],
        },
    },
}


FINAL_RUN_SQL_TOOL = {
    "type": "function",
    "function": {
        "name": ToolName.RUN_SQL,
        "description": (
            "Check a read-only query for unknown tables and columns, then run it "
            "and return the first rows. Set final=true on the query that answers "
            "the question: if it runs cleanly it is your answer and you are done."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "One SELECT query."},
                "final": {
                    "type": "boolean",
                    "description": "True when this query is your answer.",
                },
                "response": SUBMIT_TOOL["function"]["parameters"]["properties"][
                    "response"
                ],
                "assumptions": SUBMIT_TOOL["function"]["parameters"]["properties"][
                    "assumptions"
                ],
            },
            "required": ["sql"],
        },
    },
}


def _submit_with_optional_sql_tool() -> dict:
    """``submit`` with sql optional, so an information question can be answered in words."""
    function = SUBMIT_TOOL["function"]
    parameters = function["parameters"]
    properties = dict(parameters["properties"])
    properties["sql"] = {
        "type": "string",
        "description": (
            "The final SELECT query. Empty only for an information question, "
            "as the system prompt defines it."
        ),
    }
    return {
        "type": "function",
        "function": {
            **function,
            "parameters": {
                **parameters,
                "properties": properties,
                "required": ["response"],
            },
        },
    }


SUBMIT_WITH_OPTIONAL_SQL_TOOL = _submit_with_optional_sql_tool()


def _retrieve_with_purpose_tool() -> dict:
    """``retrieve`` with ``purpose``: each call picks the sql or information view."""
    function = RETRIEVE_TOOL["function"]
    parameters = function["parameters"]
    purpose = {
        "type": "string",
        "enum": list(PURPOSES),
        "description": (
            "sql (the default) shows what was found as the tables and columns to "
            "query. information describes what the catalog holds, for an "
            "information question: tables, columns, terms, attributes and "
            "definitions; with an empty query it lists every table, or every "
            'term with kinds=["term"].'
        ),
    }
    return {
        "type": "function",
        "function": {
            **function,
            "parameters": {
                **parameters,
                "properties": {**parameters["properties"], "purpose": purpose},
            },
        },
    }


RETRIEVE_WITH_PURPOSE_TOOL = _retrieve_with_purpose_tool()


def information_answers_offered(information_answers: bool) -> bool:
    """Whether information answers are offered: only through ``submit``."""
    return information_answers and config.submit_tool_enabled()


def tools_for(
    mode: EpisodeMode, *, last_turn: bool = False, information_answers: bool = False
) -> list[dict]:
    """The tools offered in *mode*; on the last turn, only the ones that answer."""
    if mode == EpisodeMode.PREDICT:
        return [PREDICT_TOOL] if last_turn else [RETRIEVE_TOOL, PREDICT_TOOL]
    if config.submit_tool_enabled():
        submit = SUBMIT_WITH_OPTIONAL_SQL_TOOL if information_answers else SUBMIT_TOOL
        retrieve = RETRIEVE_WITH_PURPOSE_TOOL if information_answers else RETRIEVE_TOOL
        answer, sql_tools = submit, [retrieve, RUN_SQL_TOOL, submit]
    else:
        answer, sql_tools = FINAL_RUN_SQL_TOOL, [RETRIEVE_TOOL, FINAL_RUN_SQL_TOOL]
    if config.grep_enabled():
        sql_tools.insert(2, GREP_TOOL)
    if mode == EpisodeMode.AUTO:
        return [answer, PREDICT_TOOL] if last_turn else [*sql_tools, PREDICT_TOOL]
    return [answer] if last_turn else sql_tools


def system_prompt_for(
    mode: EpisodeMode, dialect: str | None, *, information_answers: bool = False
) -> str:
    if mode == EpisodeMode.PREDICT:
        return PREDICT_SYSTEM_PROMPT
    lines = _SUBMIT_LINES if config.submit_tool_enabled() else _FINAL_RUN_LINES
    if config.grep_enabled():
        lines += _GREP_LINE
    opening = (
        _INFORMATION_OPENING
        if information_answers_offered(information_answers)
        else _SQL_OPENING
    )
    dialect = dialect or "SQL"
    prompt = SYSTEM_PROMPT.replace(_SUBMIT_LINES, lines, 1).format(
        opening=opening.format(dialect=dialect), dialect=dialect
    )
    return prompt + _PREDICT_WHEN if mode == EpisodeMode.AUTO else prompt


def is_answer(name: str, args: dict) -> bool:
    """Whether this call answers: ``submit``, ``predict``, or, without a submit
    tool, ``run_sql(final=true)``."""
    if name == ToolName.RUN_SQL and not config.submit_tool_enabled():
        return str(args.get("final")).lower() == "true"
    return name in ANSWERING_TOOLS


def last_call_for(mode: EpisodeMode, *, information_answers: bool = False) -> str:
    if mode == EpisodeMode.PREDICT:
        return PREDICT_LAST_CALL
    if information_answers_offered(information_answers):
        return INFORMATION_LAST_CALL
    return LAST_CALL if config.submit_tool_enabled() else FINAL_RUN_LAST_CALL


def nudge_for(mode: EpisodeMode, *, information_answers: bool = False) -> str:
    if mode == EpisodeMode.PREDICT:
        return "Call a tool: retrieve, or predict once the tables are in view."
    final = (
        "submit your final query"
        if config.submit_tool_enabled()
        else "run_sql with final=true on your final query"
    )
    if mode == EpisodeMode.AUTO:
        return f"Call a tool: retrieve, run_sql, {final}, or predict."
    if information_answers_offered(information_answers):
        return INFORMATION_NUDGE
    return NUDGE if config.submit_tool_enabled() else FINAL_RUN_NUDGE


def out_of_calls_for(mode: EpisodeMode) -> str:
    if mode == EpisodeMode.PREDICT:
        return "No tool calls left; predict."
    final = (
        "submit"
        if config.submit_tool_enabled()
        else "run your best query with final=true"
    )
    if mode == EpisodeMode.AUTO:
        return f"No tool calls left; {final} or predict."
    return f"No tool calls left; {final}."


def user_message(question: str, evidence: str | None) -> str:
    evidence = (evidence or "").strip()
    evidence_block = f"\nEVIDENCE\n{evidence}\n" if evidence else ""
    return USER_TEMPLATE.format(
        question=question.strip(), evidence_block=evidence_block
    )
