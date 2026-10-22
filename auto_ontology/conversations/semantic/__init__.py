# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A transcript's graph: what it is about, and how the statements stand.

Ingest puts a conversation's turns in the store and one vector per turn in
``vdb.conversation``. That answers a question whose wording resembles a turn's
wording. This package answers the rest.

    from auto_ontology.conversations.semantic import (
        populate_conversation_graph,
    )

    populate_conversation_graph(
        database_name="beam_100K_2", conversation_id="2"
    )

Two layers, built in that order. The **knowledge** layer is the graph an
entity-relationship extraction produces: things, and typed relations between
them in the transcript's own verbs. The **semantic** layer sits on top and says
how two *statements* stand to each other -- which value was revised, which pair
was never reconciled, which event came first. The first says what the world is;
the second says what to believe about it now.

Five modules:

``extract``
    Nodes and ``knowledge`` edges, read window by window and grounded in the
    turns a model cited. A node is a thing (``object``) or a statement
    (``fact`` / ``preference`` / ``event`` / ``state``, possibly several).
``derive``
    The ``semantic`` edges that follow from the rows: a revised value, an
    ordering of dated events. No model call.
``relate``
    The ``semantic`` edges that need reading -- ``refines``, and disagreements
    spanning different properties -- adjudicated per candidate pair.
``store``
    Reads and writes over ``conversation_node`` and ``conversation_edge``.
``note``
    A retrieved subgraph rendered as the prose block that goes in front of a
    context.

Retrieval through all of it lives in
:mod:`auto_ontology.retrieval.conversation.semantic_search`, beside the turn
search it unions with.

Imports are lazy, as the parent package's are: ``pipeline`` pulls in the DAL,
the embed client and the LLM client, and a caller that only wants
:func:`~auto_ontology.conversations.semantic.note.note_text` to render rows it
already has should not pay for a database connection it never opens.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from auto_ontology.conversations.semantic.note import note_text
    from auto_ontology.conversations.semantic.pipeline import (
        GraphResult,
        populate_conversation_graph,
        relate_conversation,
    )
    from auto_ontology.conversations.semantic.windows import Window, build_windows

__all__ = [
    "GraphResult",
    "Window",
    "build_windows",
    "note_text",
    "populate_conversation_graph",
    "relate_conversation",
]

_LAZY = {
    "GraphResult": "auto_ontology.conversations.semantic.pipeline",
    "Window": "auto_ontology.conversations.semantic.windows",
    "build_windows": "auto_ontology.conversations.semantic.windows",
    "note_text": "auto_ontology.conversations.semantic.note",
    "populate_conversation_graph": "auto_ontology.conversations.semantic.pipeline",
    "relate_conversation": "auto_ontology.conversations.semantic.pipeline",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module_name), name)
