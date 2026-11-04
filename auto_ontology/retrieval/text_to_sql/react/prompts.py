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

USER_TEMPLATE = """\
QUESTION
{question}
{evidence_block}"""

NUDGE = "Call a tool: retrieve, run_sql, or submit your final query."

LAST_CALL = "No tool calls left. Call submit now with your best query."

_KINDS = sorted(KIND_LABELS)

RETRIEVE_TOOL = {
    "type": "function",
    "function": {
        "name": "retrieve",
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
        "name": "run_sql",
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
        "name": "submit",
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

TOOLS = [RETRIEVE_TOOL, RUN_SQL_TOOL, SUBMIT_TOOL]


def user_message(question: str, evidence: str | None) -> str:
    evidence = (evidence or "").strip()
    evidence_block = f"\nEVIDENCE\n{evidence}\n" if evidence else ""
    return USER_TEMPLATE.format(
        question=question.strip(), evidence_block=evidence_block
    )
