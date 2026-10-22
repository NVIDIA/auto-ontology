# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep one transcript's extraction draft on disk, so a later stage can be retried.

Extraction is one model call per window and the better part of an hour on a long
transcript. Everything after it -- consolidation, the subject pass, the key
reconciliation, derivation, adjudication -- is minutes, and is where the tuning
happens. Twice now a bug in one of those stages has thrown away a complete
extraction that was perfectly good: once on a schema constraint, once on a
label lookup that rejected all 412 answers it was given.

The draft is the natural save point. It is the output of the expensive stage and
the input to every cheap one, it is pure data, and it is the last moment before
anything is keyed by a database id.

**The fingerprint is the whole design.** A cache that returns a draft extracted
under a prompt that has since been edited is worse than no cache: the edit
appears to do nothing, and the next hour goes into wondering why. So the key
covers everything that could change what extraction produces -- the turn text,
the window boundaries, the system prompt and the response schema -- and any
change to any of them is a miss. Being wrong in this direction costs an hour;
being wrong in the other costs a day of chasing a prompt that was never run.

Opt-in, and never on in a measured run. A cached draft is a claim about what the
current code does, and a measurement wants that claim tested rather than
assumed.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from auto_ontology.conversations.semantic.extract import GraphDraft
    from auto_ontology.conversations.semantic.windows import Window

logger = logging.getLogger(__name__)

__all__ = ["fingerprint", "load_draft", "save_draft"]

#: Bumped when the serialisation below changes shape rather than the extraction.
#: An old file then misses instead of loading into the wrong fields.
_FORMAT = 2


def fingerprint(windows: Sequence[Window]) -> str:
    """A digest of everything that decides what extraction produces.

    The turn text and the window boundaries, because they are the input; the
    system prompt and the response schema, because they are the instruction. A
    change to any of them has to miss -- see the module docstring for why that
    direction is the cheap one to be wrong in.

    Not the model name or temperature, which are read from the environment at
    call time and are not visible here. A cache hit across two models is the one
    gap left, and it is why this is opt-in.
    """
    from auto_ontology.conversations.semantic.extract import (
        _SYSTEM,
        GraphExtractionResult,
    )

    digest = hashlib.sha256()
    digest.update(f"{_FORMAT}\n".encode())
    digest.update(_SYSTEM.encode())
    digest.update(
        json.dumps(GraphExtractionResult.model_json_schema(), sort_keys=True).encode()
    )
    for window in windows:
        digest.update(repr(window.turn_indexes).encode())
        for text in window.texts:
            digest.update(text.encode())
    return digest.hexdigest()


def _node_to_json(node: Any) -> dict[str, Any]:
    """One node as plain data.

    ``vector`` is left out: extraction no longer embeds as it goes, so it is
    null here, and caching an embedding would make the file large for a value
    the next stage fills in anyway.
    """
    return {
        "name": node.name,
        "type": node.type,
        "kinds": list(node.kinds),
        "description": node.description,
        "key": node.key,
        "value": node.value,
        "occurred_on": node.occurred_on.isoformat() if node.occurred_on else None,
        "turn_indexes": sorted(node.turn_indexes),
        "verified_turn_indexes": sorted(node.verified_turn_indexes),
        "evidence": node.evidence,
        "time_anchor": node.time_anchor,
    }


def _edge_to_json(edge: Any) -> dict[str, Any]:
    return {
        "source": edge.source,
        "target": edge.target,
        "role": edge.role,
        "turn_indexes": sorted(edge.turn_indexes),
        "verified_turn_indexes": sorted(edge.verified_turn_indexes),
        "evidence": edge.evidence,
    }


def save_draft(path: Path, digest: str, graph: GraphDraft) -> None:
    """Write *graph* to *path*, stamped with *digest*.

    Written to a neighbouring temporary file and moved into place, so an
    interrupted write cannot leave a truncated draft that the next run would
    read as a complete one.

    Failure is logged, not raised. The draft is in memory and the build can
    finish without ever writing it; losing the cache costs the next run an hour
    and this run nothing.
    """
    payload = {
        "fingerprint": digest,
        "nodes": [_node_to_json(node) for node in graph.nodes],
        "edges": [_edge_to_json(edge) for edge in graph.edges],
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".partial")
        temporary.write_text(json.dumps(payload))
        temporary.replace(path)
    except Exception:
        logger.exception(
            "Could not cache the extraction draft to %s -- a rerun will re-extract.",
            path,
        )
        return
    logger.info(
        "Cached %d node(s) and %d edge(s) to %s.",
        len(graph.nodes),
        len(graph.edges),
        path,
    )


def load_draft(path: Path, digest: str) -> GraphDraft | None:
    """Read the draft at *path* if it was extracted under *digest*, else nothing.

    A mismatch is reported at INFO and is not an error: it is what should happen
    after a prompt edit, and a run that re-extracts is a run that is measuring
    the code it is actually running.
    """
    from auto_ontology.conversations.semantic.extract import (
        EdgeDraft,
        GraphDraft,
        NodeDraft,
    )

    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception:
        logger.exception(
            "Could not read the cached draft at %s -- re-extracting.", path
        )
        return None

    if payload.get("fingerprint") != digest:
        logger.info(
            "The cached draft at %s was extracted under a different prompt, "
            "schema or transcript -- re-extracting.",
            path,
        )
        return None

    try:
        graph = GraphDraft(
            nodes=[
                NodeDraft(
                    name=row["name"],
                    type=row["type"],
                    kinds=list(row["kinds"]),
                    description=row["description"],
                    key=row["key"],
                    value=row["value"],
                    occurred_on=(
                        date.fromisoformat(row["occurred_on"])
                        if row["occurred_on"]
                        else None
                    ),
                    turn_indexes=set(row["turn_indexes"]),
                    verified_turn_indexes=set(row["verified_turn_indexes"]),
                    evidence=row["evidence"],
                    time_anchor=row["time_anchor"],
                )
                for row in payload["nodes"]
            ],
            edges=[
                EdgeDraft(
                    source=row["source"],
                    target=row["target"],
                    role=row["role"],
                    turn_indexes=set(row["turn_indexes"]),
                    verified_turn_indexes=set(row["verified_turn_indexes"]),
                    evidence=row["evidence"],
                )
                for row in payload["edges"]
            ],
        )
    except Exception:
        logger.exception("The cached draft at %s does not load -- re-extracting.", path)
        return None

    logger.info(
        "Reusing %d cached node(s) and %d edge(s) from %s -- skipping extraction.",
        len(graph.nodes),
        len(graph.edges),
        path,
    )
    return graph
