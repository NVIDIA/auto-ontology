# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Re-embedding the attributes of columns whose values were just withheld."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from auto_ontology.semantic.pii_values import withhold_pii_column_values

_MODULE = "auto_ontology.semantic.pii_values"

_TERMS = [
    {"id": "term-user", "name": "User", "synonyms": ["Account"]},
    {"id": "term-order", "name": "Order", "synonyms": []},
]
_ATTRS = [
    {"id": "a-email", "source_column": "email", "term_name": "User"},
    {"id": "a-status", "source_column": "status", "term_name": "Order"},
]


def _withhold(cleared: list[dict], embedder: MagicMock | None) -> tuple[int, MagicMock]:
    with (
        patch(f"{_MODULE}.clear_values_of_pii_columns", return_value=cleared),
        patch(
            f"{_MODULE}.fetch_terms_and_attributes_for_table",
            return_value=(_TERMS, _ATTRS),
        ),
        patch(f"{_MODULE}.build_semantic_embedder", return_value=embedder) as build,
    ):
        return withhold_pii_column_values(), build


def test_only_the_cleared_columns_attributes_are_re_embedded() -> None:
    embedder = MagicMock()
    cleared = [{"column_name": "email", "table_id": "t1", "database_name": "db"}]

    count, build = _withhold(cleared, embedder)

    assert count == 1
    build.assert_called_once_with("db", reset=False)
    embedder.vdb.delete_by_id.assert_called_once_with("a-email")
    embedder.embed_term.assert_called_once_with({"synonyms": ["Account"]}, [_ATTRS[0]])


def test_one_embedder_per_database() -> None:
    embedder = MagicMock()
    cleared = [
        {"column_name": "email", "table_id": "t1", "database_name": "db"},
        {"column_name": "status", "table_id": "t2", "database_name": "db"},
    ]

    _, build = _withhold(cleared, embedder)

    build.assert_called_once()
    assert embedder.embed_term.call_count == 2


def test_nothing_cleared_touches_no_vectors() -> None:
    count, build = _withhold([], MagicMock())

    assert count == 0
    build.assert_not_called()


def test_without_an_embedder_the_catalog_is_still_cleared() -> None:
    cleared = [{"column_name": "email", "table_id": "t1", "database_name": "db"}]

    count, _ = _withhold(cleared, None)

    assert count == 1


def test_a_failed_re_embed_does_not_stop_the_other_tables() -> None:
    embedder = MagicMock()
    embedder.embed_term.side_effect = [RuntimeError("embed down"), 1]
    cleared = [
        {"column_name": "email", "table_id": "t1", "database_name": "db"},
        {"column_name": "status", "table_id": "t2", "database_name": "db"},
    ]

    count, _ = _withhold(cleared, embedder)

    assert count == 2
    assert embedder.embed_term.call_count == 2
