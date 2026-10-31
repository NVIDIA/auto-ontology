# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A table that stores links in both directions is named to every SQL writer."""

from __future__ import annotations

import pytest

from auto_ontology.retrieval.text_to_sql.agents import candidates_preparation as cp
from auto_ontology.retrieval.text_to_sql.agents.sql_reconstruction import (
    _format_known_columns,
)

_CONNECTED = {
    "id": "t-connected",
    "name": "connected",
    "symmetric_column_pairs": [["atom_id", "atom_id2"]],
    "columns": [],
}


def test_a_repair_is_told_not_to_join_both_link_columns() -> None:
    rendered = _format_known_columns(None, None, [_CONNECTED, {"name": "atom"}])

    assert "SYMMETRIC LINK TABLES" in rendered
    assert "- `connected` stores every (atom_id, atom_id2) link" in rendered


def test_no_symmetric_table_leaves_the_repair_context_unchanged() -> None:
    assert _format_known_columns(None, None, [{"name": "atom"}]) == ""
    assert _format_known_columns(None, None) == ""


def test_candidate_tables_get_their_pairs_from_the_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cp,
        "fetch_symmetric_column_pairs",
        lambda ids: {"t-connected": [["atom_id", "atom_id2"]]},
    )
    tables = [{"id": "t-connected", "name": "connected"}, {"id": "t-atom"}]

    cp._attach_symmetric_column_pairs(tables)

    assert tables[0]["symmetric_column_pairs"] == [["atom_id", "atom_id2"]]
    assert "symmetric_column_pairs" not in tables[1]


def test_a_failed_read_leaves_the_candidates_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail(ids):
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(cp, "fetch_symmetric_column_pairs", _fail)
    tables = [{"id": "t-connected", "name": "connected"}]

    cp._attach_symmetric_column_pairs(tables)

    assert tables == [{"id": "t-connected", "name": "connected"}]
