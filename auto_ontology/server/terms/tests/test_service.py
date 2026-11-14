# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Term service orchestration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from auto_ontology.server.terms import service


@patch("auto_ontology.dal.attributes.update_column_attribute")
@patch("auto_ontology.dal.attributes.fetch_column_attribute_column")
def test_sample_values_are_refused_before_any_metadata_is_written(
    mock_fetch_column: MagicMock,
    mock_update: MagicMock,
) -> None:
    """A refused sample value must not leave a renamed attribute behind.

    The name is written by a different Neo4j statement than the samples, so
    checking the Column's type only at the sample write would commit the
    rename and then answer 422.
    """
    mock_fetch_column.return_value = {"id": "col-1", "data_type": "integer"}

    with pytest.raises(ValueError, match="integer"):
        service.update_column_attribute(
            "term-1",
            "attr-1",
            name="Renamed",
            sample_values=["abc"],
        )

    mock_update.assert_not_called()


@patch("auto_ontology.dal.attributes.update_column_attribute")
@patch("auto_ontology.dal.attributes.fetch_column_attribute_column")
def test_a_missing_attribute_still_answers_not_found(
    mock_fetch_column: MagicMock,
    mock_update: MagicMock,
) -> None:
    """With no Column to read a type from, the write decides — a 404, not a 422."""
    mock_fetch_column.return_value = None
    mock_update.return_value = None

    assert (
        service.update_column_attribute("term-1", "attr-1", sample_values=["abc"])
        is None
    )


def _updated_row(**extra: object) -> dict[str, object]:
    return {
        "id": "attr-1",
        "name": "Total",
        "description": "Long attribute description",
        "column_description": "Order total",
        "family_prefix": "NEW_",
        "term_name": "Order",
        "source_column": "total",
        "sample_values": "[10]",
        "database_name": "shop",
        **extra,
    }


@patch.object(service, "build_semantic_embedder")
@patch("auto_ontology.dal.attributes.update_column_attribute")
def test_a_description_edit_does_not_re_embed_the_attribute(
    mock_update: MagicMock,
    mock_embedder: MagicMock,
) -> None:
    mock_update.return_value = _updated_row()

    row = service.update_column_attribute("term-1", "attr-1", description="New text")

    assert row["sample_values"] == ["10"]
    mock_embedder.assert_not_called()


@patch.object(service, "build_semantic_embedder")
@patch("auto_ontology.dal.attributes.update_column_attribute")
def test_a_rename_re_embeds_from_the_column_description(
    mock_update: MagicMock,
    mock_embedder: MagicMock,
) -> None:
    mock_update.return_value = _updated_row(name="Order Total")
    embedder = mock_embedder.return_value

    service.update_column_attribute("term-1", "attr-1", name="Order Total")

    embedder.vdb.delete_by_id.assert_called_once_with("attr-1")
    (_, (attr,)) = embedder.embed_term.call_args.args
    assert attr["column_description"] == "Order total"
    assert attr["family_prefix"] == "NEW_"
    assert "description" not in attr
