# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conversational memory: ingest a chat transcript the way a database is.

The tabular path ingests a schema -- tables and columns -- and embeds one row
per catalog node into ``vdb.data_objects_layer``. This package ingests a
transcript and embeds one row per question/answer pair into ``vdb.conversation``,
under the same ``database_name`` scope. Same machinery, same embed model, same
row shape; a separate collection, so a catalog search and a transcript search
cannot return each other's hits.

    from auto_ontology.conversations import (
        Conversation,
        run_conversation_ingest,
        turns_from_sessions,
    )

    conversation = Conversation(
        conversation_id="42",
        turns=tuple(turns_from_sessions(sessions)),
    )
    run_conversation_ingest(conversation, database_name="beam_100K_42")

Imports are lazy. ``ingest`` pulls in pandas, the DAL and the embed client,
and the parsing half of this package is useful without any of them -- a
caller turning an export into :class:`Turn` objects should not pay for a
database connection it never opens.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from auto_ontology.conversations.model import (
    ASSISTANT_ROLES,
    USER_ROLES,
    Conversation,
    Turn,
    turns_from_messages,
    turns_from_sessions,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from auto_ontology.conversations.embedding_rows import (
        ConversationEmbeddingRowsOp,
        build_turn_text,
    )
    from auto_ontology.conversations.ingest import (
        ingest_conversation,
        run_conversation_ingest,
    )

__all__ = [
    "ASSISTANT_ROLES",
    "Conversation",
    "ConversationEmbeddingRowsOp",
    "Turn",
    "USER_ROLES",
    "build_turn_text",
    "ingest_conversation",
    "run_conversation_ingest",
    "turns_from_messages",
    "turns_from_sessions",
]

_LAZY = {
    "ConversationEmbeddingRowsOp": "auto_ontology.conversations.embedding_rows",
    "build_turn_text": "auto_ontology.conversations.embedding_rows",
    "ingest_conversation": "auto_ontology.conversations.ingest",
    "run_conversation_ingest": "auto_ontology.conversations.ingest",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module_name), name)
