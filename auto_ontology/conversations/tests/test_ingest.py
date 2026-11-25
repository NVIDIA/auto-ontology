# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`auto_ontology.conversations.ingest`.

The store and the embed endpoint are both stubbed: what is under test is the
contract between the steps -- that every turn is upserted under the database,
that the ids come back on the frame the embed step reads, and that the
pgvector reset cannot fire before there is something to write.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterator
from unittest.mock import patch

import pandas as pd
import pytest

from auto_ontology.catalog.constants import Labels
from auto_ontology.conversations.ingest import (
    ingest_conversation,
    run_conversation_ingest,
)
from auto_ontology.conversations.model import Conversation, Turn


@contextlib.contextmanager
def _no_transaction() -> Iterator[None]:
    yield


def _conversation(n: int = 2) -> Conversation:
    return Conversation(
        conversation_id="42",
        turns=tuple(
            Turn(
                question=f"q{i}",
                answer=f"a{i}",
                index=i,
                session_index=0,
                time_anchor="March-15-2024",
            )
            for i in range(n)
        ),
    )


@pytest.fixture
def upserts() -> Iterator[list[tuple[Any, ...]]]:
    """Record every upsert, returning a distinct id per call."""
    calls: list[tuple[Any, ...]] = []

    def _upsert(label: str, match: dict, values: dict, **kwargs: Any) -> str:
        calls.append((label, match, values, kwargs))
        if label == Labels.DB:
            return "db-1"
        return f"turn-{match['turn_index']}"

    with (
        patch("auto_ontology.conversations.ingest.upsert_row", side_effect=_upsert),
        patch("auto_ontology.conversations.ingest.write_transaction", _no_transaction),
        patch(
            "auto_ontology.conversations.ingest._delete_turns_not_in",
            return_value=[],
        ),
    ):
        yield calls


def test_turns_the_new_transcript_lacks_are_deleted(upserts: list) -> None:
    """A shorter re-ingest must not leave the old tail behind."""
    with patch(
        "auto_ontology.conversations.ingest._delete_turns_not_in", return_value=[]
    ) as delete:
        ingest_conversation(_conversation(2), database_name="beam_42")

    assert delete.call_args.kwargs == {
        "database_id": "db-1",
        "conversation_id": "42",
        "keep": [0, 1],
    }


def test_the_vectors_of_deleted_turns_are_dropped(upserts: list) -> None:
    vdb = _FakeVdb()
    with (
        patch(
            "auto_ontology.conversations.ingest._delete_turns_not_in",
            return_value=["turn-2", "turn-3"],
        ),
        patch(
            "auto_ontology.conversations.ingest.get_conversation_vdb",
            return_value=vdb,
        ) as get_vdb,
    ):
        ingest_conversation(_conversation(2), database_name="shared")

    assert get_vdb.call_args.kwargs == {"database_name": "shared"}
    assert vdb.deleted == ["turn-2", "turn-3"]


def test_nothing_removed_leaves_the_vectors_alone(upserts: list) -> None:
    with patch("auto_ontology.conversations.ingest.get_conversation_vdb") as get_vdb:
        ingest_conversation(_conversation(2), database_name="shared")

    get_vdb.assert_not_called()


def test_writes_the_database_then_every_turn(upserts: list) -> None:
    frame = ingest_conversation(_conversation(2), database_name="beam_42")

    assert [call[0] for call in upserts] == [
        Labels.DB,
        Labels.CONVERSATION,
        Labels.CONVERSATION,
    ]
    assert list(frame["id"]) == ["turn-0", "turn-1"]
    assert list(frame["question"]) == ["q0", "q1"]


def test_every_turn_hangs_off_the_database(upserts: list) -> None:
    """Without the parent id the row fails its NOT NULL, not its test."""
    ingest_conversation(_conversation(2), database_name="beam_42")

    for label, _match, _values, kwargs in upserts:
        if label == Labels.CONVERSATION:
            assert kwargs["parent_id"] == "db-1"


def test_upsert_is_keyed_on_the_turn_position(upserts: list) -> None:
    ingest_conversation(_conversation(1), database_name="beam_42")

    _label, match, _values, _kwargs = upserts[1]
    assert match == {"conversation_id": "42", "turn_index": 0}


def test_an_existing_turn_is_updated_not_left_alone(upserts: list) -> None:
    """An edited transcript must change the stored text, not just resolve it."""
    ingest_conversation(_conversation(1), database_name="beam_42")

    _label, _match, values, kwargs = upserts[1]
    assert kwargs["on_match"] == values
    assert values["question"] == "q0"


def test_a_conversation_with_no_turns_writes_nothing(upserts: list) -> None:
    frame = ingest_conversation(
        Conversation(conversation_id="42"), database_name="beam_42"
    )

    assert upserts == []
    assert frame.empty
    assert "question" in frame.columns


def test_run_ingest_embeds_and_writes(upserts: list) -> None:
    vdb = _FakeVdb(written=3)
    with _embedding([pd.DataFrame([{"text": "t"}])], vdb) as get_vdb:
        written = run_conversation_ingest(_conversation(2), database_name="beam_42")

    assert written == 3
    assert get_vdb.call_args.kwargs == {"database_name": "beam_42"}
    # Under each turn's id, so a re-ingest replaces rather than adds a copy.
    assert vdb.upserts == [True]


def test_reset_clears_only_this_conversation(upserts: list) -> None:
    """Others sharing the database keep their vectors."""
    vdb = _FakeVdb()
    with _embedding([pd.DataFrame([{"text": "t"}])], vdb):
        run_conversation_ingest(_conversation(1), database_name="shared")

    assert vdb.cleared == [("shared", "42")]


def test_reset_can_be_turned_off(upserts: list) -> None:
    vdb = _FakeVdb()
    with _embedding([pd.DataFrame([{"text": "t"}])], vdb):
        run_conversation_ingest(_conversation(1), database_name="shared", reset=False)

    assert vdb.cleared == []
    assert vdb.upserts == [True]


def test_pgvector_is_untouched_when_nothing_embeds(upserts: list) -> None:
    """Resetting before the first usable chunk would delete the live index."""
    with _embedding([pd.DataFrame([{"text": "t"}])], _FakeVdb(), records=[]) as get_vdb:
        written = run_conversation_ingest(_conversation(2), database_name="beam_42")

    assert written == 0
    get_vdb.assert_not_called()


def test_a_blank_database_name_is_refused(upserts: list) -> None:
    with pytest.raises(ValueError, match="database_name"):
        run_conversation_ingest(_conversation(1), database_name="   ")


class _FakeVdb:
    def __init__(self, written: int = 0) -> None:
        self._written = written
        self.deleted: list[str] = []
        self.cleared: list[tuple[str, str]] = []
        self.upserts: list[bool] = []

    def run(self, records: Any, *, upsert: bool = False) -> int:
        self.upserts.append(upsert)
        return self._written

    def delete_by_conversation(self, database_name: str, conversation_id: str) -> int:
        self.cleared.append((database_name, conversation_id))
        return 1

    def delete_by_id(self, node_id: str) -> int:
        self.deleted.append(node_id)
        return 1


@contextlib.contextmanager
def _embedding(
    chunks: list[pd.DataFrame], vdb: _FakeVdb, records: Any = ("r",)
) -> Iterator[Any]:
    with (
        patch("auto_ontology.conversations.ingest.get_embed_params"),
        patch(
            "auto_ontology.conversations.ingest.batch_embed_chunks",
            return_value=iter(chunks),
        ),
        patch(
            "auto_ontology.conversations.ingest.to_vdb_records",
            return_value=list(records),
        ),
        patch(
            "auto_ontology.conversations.ingest.get_conversation_vdb", return_value=vdb
        ) as get_vdb,
    ):
        yield get_vdb
