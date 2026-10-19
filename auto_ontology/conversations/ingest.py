# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The conversation write path -- the chat counterpart of tabular ingest.

Same three straight-line steps as
:func:`auto_ontology.ingestion_service.ingest.run_ingest`, against a different
source: the store write (:func:`ingest_conversation`), the embedding rows
(:class:`~auto_ontology.conversations.embedding_rows.ConversationEmbeddingRowsOp`),
and the embed (:func:`auto_ontology.utils.embedding.batch_embed_chunks`).

It takes a :class:`~auto_ontology.conversations.model.Conversation` rather than
a connector. Nothing about a transcript is introspectable the way a schema is
-- there are no tables to list, no types to read, no rows to sample -- so the
extraction half of the tabular path has no counterpart here and the caller
hands over the parsed turns directly.

**``database_name`` is the memory scope, not a database.** Conversational
memory is built per transcript: what the assistant may recall when answering
about one chat is that chat, and letting a vector search cross into another
user's history is a leak rather than a feature. Giving each conversation its
own ``database_name`` buys that isolation from machinery that already exists
-- pgvector rows are filtered by it, and ``delete_all_data`` is scoped to it.
Passing one name for several conversations is supported and pools them;
``conversation_id`` still keeps the transcripts apart on the rows.
"""

from __future__ import annotations

import logging
import time

import pandas as pd

from auto_ontology.catalog.constants import Labels
from auto_ontology.catalog.store.rows import upsert_row
from auto_ontology.conversations.embedding_rows import ConversationEmbeddingRowsOp
from auto_ontology.conversations.model import Conversation
from auto_ontology.dal.session import write_transaction
from auto_ontology.utils import get_embed_params
from auto_ontology.utils.embedding import batch_embed_chunks
from auto_ontology.vdb import get_conversation_vdb
from auto_ontology.vdb.records import to_vdb_records

logger = logging.getLogger(__name__)

__all__ = ["ingest_conversation", "run_conversation_ingest"]

#: Columns on the DataFrame :func:`ingest_conversation` returns. Named once so
#: the empty case has the same shape as the populated one -- a caller that
#: indexes a column should not have to care which it got.
TURN_COLUMNS = [
    "id",
    "conversation_id",
    "turn_index",
    "session_index",
    "question",
    "answer",
    "time_anchor",
]


def ingest_conversation(
    conversation: Conversation, *, database_name: str
) -> pd.DataFrame:
    """Write *conversation*'s turns to the store, return what landed.

    The returned DataFrame carries the id assigned to each turn, so the embed
    step can build its text without a round trip back to the store -- the same
    contract :func:`auto_ontology.catalog.ingest_catalog` has.

    Upserts rather than inserts, keyed on ``(conversation_id, turn_index)``
    within the database. Re-ingesting an edited transcript updates the turns
    in place; inserting would leave the previous run's rows behind, and a
    retrieval that returns both copies of an exchange looks exactly like a
    conversation in which it happened twice.

    The whole transcript is written in one transaction. A half-written
    conversation is worse than none: the gap is invisible to retrieval, which
    returns the turns that made it and nothing to say others are missing.
    """
    if not conversation.turns:
        return pd.DataFrame(columns=TURN_COLUMNS)

    rows: list[dict[str, object]] = []
    with write_transaction():
        database_id = upsert_row(
            Labels.DB, {"name": database_name}, {"name": database_name}
        )
        for turn in conversation.turns:
            match = {
                "conversation_id": conversation.conversation_id,
                "turn_index": turn.index,
            }
            values = {
                **match,
                "session_index": turn.session_index,
                "question": turn.question,
                "answer": turn.answer,
                "time_anchor": turn.time_anchor,
            }
            turn_id = upsert_row(
                Labels.CONVERSATION,
                match,
                values,
                parent_id=database_id,
                # The text is the whole row. Without on_match an upsert of an
                # existing turn resolves its id and writes nothing, so an
                # edited transcript would report success and change nothing.
                on_match=values,
            )
            rows.append({"id": turn_id, **values})

    return pd.DataFrame(rows, columns=TURN_COLUMNS)


def run_conversation_ingest(
    conversation: Conversation,
    *,
    database_name: str,
    reset: bool = True,
) -> int:
    """Write, embed and index one conversation. Returns the rows written.

    *reset* clears the database's existing vectors before the first write, as
    tabular ingest does, so a re-ingest replaces rather than accumulates. Turn
    it off when several conversations share one ``database_name`` -- otherwise
    each one wipes the last.
    """
    database_name = database_name.strip()
    if not database_name:
        raise ValueError("database_name is required: it is what scopes the memory")

    started = time.monotonic()
    logger.info(
        "Conversation ingestion started for %s (database %s)",
        conversation.conversation_id,
        database_name,
    )

    turns_df = ingest_conversation(conversation, database_name=database_name)
    embed_rows = ConversationEmbeddingRowsOp(database_name=database_name)(turns_df)

    vdb = None
    rows_written = 0
    for chunk in batch_embed_chunks(
        embed_rows, get_embed_params(), label=database_name
    ):
        records = to_vdb_records(chunk)
        if not records:
            # Every row in the chunk came back without an embedding. Already
            # logged as a warning by the embed step; nothing to write.
            continue
        if vdb is None:
            # Reset lazily, on the first chunk that actually has something to
            # write -- resetting up front would delete the stored turns and
            # then spend minutes embedding, so a bad key or a dead endpoint
            # would leave retrieval with nothing instead of the previous run.
            vdb = get_conversation_vdb(database_name=database_name, reset=reset)
        rows_written += vdb.run(records)

    logger.info(
        "Conversation ingestion finished for %s -- %d turn(s), "
        "%d embedding row(s) in %.1fs",
        conversation.conversation_id,
        len(turns_df),
        rows_written,
        time.monotonic() - started,
    )
    return rows_written
