# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a prediction has to say about itself afterwards.

A wrong answer reported days later can only be explained if the run that
produced it can be reconstructed: which graph, built from which schema, by which
engine, under which prompt, and what query the model actually wrote. Predictions
are not deterministic and the data underneath them moves, so re-running the
question is not a way of finding out what happened the first time.

One line is emitted per prediction, whether it answered, refused, or failed. It
names the run rather than describing it: identifiers and counts, no table
contents, and not the question, which is the user's words and may carry anything
they typed.

The generated PQL is included, because it is what distinguishes a model that
misread the question from a graph that was built wrong. Its literals are not: an
entity list holds warehouse identities and a quoted filter holds whatever the
question named, so both are replaced by their shape. What is left is the
operator's own schema and the structure of the query, which is what the PQL was
being kept for.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_STRING_LITERAL = re.compile(r"'(?:[^']|'')*'")
_IN_LIST = re.compile(r"\bIN\s*\(([^()]*)\)", re.IGNORECASE)
# An equality against a bare number. A threshold is written with an inequality
# and says how big, which is query shape; an equality against a number names one
# row, which is an identity. The `=` closing a `<=` or `>=` is a threshold, not
# an equality; `!=` still names the row it excludes, so it goes like `=` does.
_EQUALS_NUMBER = re.compile(r"((?<![<>])=\s*)-?\d+(?:\.\d+)?\b")

# Bumped when a field in RunRecord is renamed or removed, so a reader can tell
# which shape it is looking at rather than discovering the change by breaking.
SCHEMA_VERSION = 1

CACHE_HIT = "hit"
CACHE_MISS = "miss"
CACHE_DISABLED = "disabled"


@dataclass(frozen=True)
class GraphIdentity:
    """Which graph a prediction ran against, and what it cost to have it.

    Carried on the built context so a prediction can name its graph without
    holding the graph itself, and so a cached one still reports the build it came
    from alongside the fact that it was reused.
    """

    fingerprint: str = ""
    engine_version: str = ""
    prompt_version: str = ""
    tables: int = 0
    rows: int = 0
    bytes: int = 0
    edges: int = 0
    edges_from: str = ""
    build_seconds: float = 0.0
    cache: str = CACHE_DISABLED

    def reused(self) -> "GraphIdentity":
        """The same graph, named as a reuse rather than as the build it came from."""
        return dataclasses.replace(self, cache=CACHE_HIT)


@dataclass
class RunRecord:
    """One prediction, as it will be read back when something looks wrong."""

    outcome: str
    schema_version: int = SCHEMA_VERSION
    llm_model: str = ""
    pql: str = ""
    attempts: int = 0
    entities: int = 0
    rows_returned: int = 0
    seconds: float = 0.0
    error: str = ""
    graph: GraphIdentity = field(default_factory=GraphIdentity)


def redact_literals(pql: str) -> str:
    """The query without the values it names, which are the warehouse's data.

    An entity list is a list of real identities and a quoted filter is whatever
    the question named, both of which the log is not the place for.

    An equality against a bare number goes too, since that names one row.
    Inequalities and aggregation arguments stay: a threshold says how big and a
    window says how long, which is the shape of the question rather than an
    answer to it, and neither names anybody.
    """
    if not pql:
        return ""
    redacted = _STRING_LITERAL.sub("'?'", pql)

    def _count(match: "re.Match[str]") -> str:
        inner = match.group(1).strip()
        values = len([v for v in inner.split(",") if v.strip()]) if inner else 0
        return f"IN ({values} values)"

    redacted = _IN_LIST.sub(_count, redacted)
    return _EQUALS_NUMBER.sub(r"\g<1>?", redacted)


def redact_error(message: str) -> str:
    """An error message with the values it quoted back removed.

    A warehouse or engine rejecting a query commonly echoes the offending
    fragment, which carries the same literals the query did. Left alone it
    would put in the log exactly what redacting the query kept out.
    """
    return redact_literals(message) if message else ""


def emit(record: RunRecord) -> None:
    """Write one run to the log, never failing the request it describes."""
    try:
        logger.info("kumo.run %s", json.dumps(asdict(record), default=str))
    except Exception:
        logger.debug("kumo: run record could not be emitted", exc_info=True)


def llm_model_name(llm: Any) -> str:
    """Which model wrote the query, under whichever name its client records it."""
    for attribute in ("model_name", "model", "model_id", "deployment_name"):
        value = getattr(llm, attribute, None)
        if isinstance(value, str) and value:
            return value
    return ""
