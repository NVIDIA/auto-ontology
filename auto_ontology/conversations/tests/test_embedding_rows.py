# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`auto_ontology.conversations.embedding_rows`.

:func:`build_turn_text` is a retrieval contract, so these assert the parts a
query is written against -- that both sides of the pair are present, that the
identifiers are in the text, and that the row shape matches what the tabular
path emits into the same collection.
"""

from __future__ import annotations

import pandas as pd

from auto_ontology.catalog.constants import Labels
from auto_ontology.conversations.embedding_rows import (
    ConversationEmbeddingRowsOp,
    build_turn_text,
)
from auto_ontology.utils.embedding_rows import CatalogEmbeddingRowsOp


def _turns_df(**overrides: object) -> pd.DataFrame:
    row: dict[str, object] = {
        "id": "turn-1",
        "conversation_id": "42",
        "turn_index": 3,
        "session_index": 1,
        "question": "when is the deadline?",
        "answer": "March 15.",
        "time_anchor": "March-01-2024",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_one_row_per_pair() -> None:
    rows = ConversationEmbeddingRowsOp(database_name="beam_42")(
        pd.concat([_turns_df(), _turns_df(id="turn-2", turn_index=4)])
    )

    assert len(rows) == 2


def test_text_carries_both_sides_and_the_identifiers() -> None:
    rows = ConversationEmbeddingRowsOp(database_name="beam_42")(_turns_df())
    text = rows.iloc[0]["text"]

    assert "when is the deadline?" in text
    assert "March 15." in text
    assert "conversation_id: 42" in text
    assert "time_anchor: March-01-2024" in text
    assert "database_name: beam_42" in text


def test_position_is_spelled_out_rather_than_left_as_a_bare_number() -> None:
    """Every turn in a sitting shares one date, so order has to be stated."""
    frame = pd.concat(
        [
            _turns_df(id="a", turn_index=0, session_index=0),
            _turns_df(id="b", turn_index=1, session_index=0),
            _turns_df(id="c", turn_index=2, session_index=1),
        ]
    )
    rows = ConversationEmbeddingRowsOp(database_name="beam_42")(frame)

    # One-based in the prose, as "of N" implies; turn_index stays zero-based.
    assert "turn 1 of 3" in rows.iloc[0]["text"]
    assert "turn 2 of 3" in rows.iloc[1]["text"]
    assert "session 1 of 2" in rows.iloc[0]["text"]
    assert "session 2 of 2" in rows.iloc[2]["text"]


def test_the_total_comes_from_the_highest_index_not_the_frame_size() -> None:
    """A partial re-embed must not render "turn 4 of 2"."""
    frame = pd.concat(
        [
            _turns_df(id="a", turn_index=3),
            _turns_df(id="b", turn_index=4),
        ]
    )
    rows = ConversationEmbeddingRowsOp(database_name="beam_42")(frame)

    assert "turn 4 of 5" in rows.iloc[0]["text"]
    assert "turn 5 of 5" in rows.iloc[1]["text"]


def test_a_later_session_is_dated_relative_to_the_first() -> None:
    """The gap between sittings is real; the turn-level clock time is not."""
    frame = pd.concat(
        [
            _turns_df(
                id="a", turn_index=0, session_index=0, time_anchor="March-15-2024"
            ),
            _turns_df(
                id="b", turn_index=1, session_index=1, time_anchor="April-05-2024"
            ),
        ]
    )
    rows = ConversationEmbeddingRowsOp(database_name="beam_42")(frame)

    assert "the day the conversation began" in rows.iloc[0]["text"]
    assert "21 days after the conversation began" in rows.iloc[1]["text"]


def test_an_unparseable_anchor_costs_only_the_day_offset() -> None:
    rows = ConversationEmbeddingRowsOp(database_name="beam_42")(
        _turns_df(time_anchor="sometime last spring")
    )
    text = rows.iloc[0]["text"]

    assert "days after" not in text
    assert "time_anchor: sometime last spring" in text


def test_row_shape_matches_the_tabular_path() -> None:
    """Both land in the same collection, so both must present the same columns."""
    conversation_rows = ConversationEmbeddingRowsOp(database_name="beam_42")(
        _turns_df()
    )
    catalog_rows = CatalogEmbeddingRowsOp(database_name="pagila")(
        (
            pd.DataFrame(
                [{"id": "t1", "table_name": "film", "table_schema": "public"}]
            ),
            pd.DataFrame(
                [
                    {
                        "id": "c1",
                        "table_name": "film",
                        "table_schema": "public",
                        "column_name": "title",
                    }
                ]
            ),
        )
    )

    assert list(conversation_rows.columns) == list(catalog_rows.columns)


def test_metadata_locates_the_turn_in_its_transcript() -> None:
    """Widening a hit to its neighbours needs the conversation and position."""
    rows = ConversationEmbeddingRowsOp(database_name="beam_42")(_turns_df())
    metadata = rows.iloc[0]["metadata"]

    assert metadata["label"] == Labels.CONVERSATION
    assert metadata["id"] == "turn-1"
    assert metadata["database_name"] == "beam_42"
    assert metadata["conversation_id"] == "42"
    assert metadata["turn_index"] == 3
    assert metadata["session_index"] == 1
    assert metadata["content_metadata"]["conversation_id"] == "42"


def test_drops_a_pair_with_nothing_said() -> None:
    rows = ConversationEmbeddingRowsOp(database_name="beam_42")(
        _turns_df(question="", answer="")
    )

    assert rows.empty
    assert list(rows.columns) == ["text", "path", "page_number", "metadata"]


def test_keeps_a_pair_with_only_one_side() -> None:
    rows = ConversationEmbeddingRowsOp(database_name="beam_42")(_turns_df(answer=""))

    assert len(rows) == 1


def test_optional_fields_are_omitted_rather_than_rendered_as_none() -> None:
    """`session: None` in the text would be embedded as if it meant something."""
    text = build_turn_text(
        question="q",
        answer="a",
        conversation_id="42",
        turn_index=0,
        turn_count=None,
        session_index=None,
        session_count=None,
        time_anchor=None,
        days_since_start=None,
        database_name="beam_42",
    )

    assert "session" not in text
    assert "time_anchor" not in text
    assert "None" not in text
    # Without a total there is no "of N", but the position still has to show.
    assert "turn 1" in text


def test_empty_frame_keeps_the_output_columns() -> None:
    rows = ConversationEmbeddingRowsOp(database_name="beam_42")(pd.DataFrame())

    assert rows.empty
    assert list(rows.columns) == ["text", "path", "page_number", "metadata"]
