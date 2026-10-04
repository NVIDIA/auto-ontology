# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for catalog node patching, including the sample-value type guard."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
from nemo_retriever.tabular_data.ingestion.model.reserved_words import Labels

from auto_ontology.server.datasources import service


def _patched_column(**props: object) -> dict[str, object]:
    return {"id": "col-1", "label": Labels.COLUMN, "props": props}


@pytest.fixture(autouse=True)
def allow_sample_data_movement() -> Iterator[MagicMock]:
    with patch.object(
        service, "is_column_id_safe_for_data_movement", return_value=True
    ) as mocked:
        yield mocked


@patch.object(service, "_refresh_semantic_column_attribute_embeddings")
@patch.object(service, "_refresh_vdb_embeddings")
@patch.object(service, "patch_catalog_node")
@patch.object(service, "fetch_node_properties_by_id")
def test_sample_values_reach_a_text_column(
    mock_fetch: MagicMock,
    mock_patch: MagicMock,
    _mock_vdb: MagicMock,
    _mock_semantic: MagicMock,
) -> None:
    mock_fetch.return_value = {"data_type": "character varying(50)"}
    mock_patch.return_value = _patched_column(sample_values=["open", "closed"])

    result = service.update_node_properties(
        "col-1", {"sample_values": ["open", "closed"]}
    )

    assert result == {"id": "col-1", "sample_values": ["open", "closed"]}
    mock_patch.assert_called_once()


@patch.object(service, "_refresh_semantic_column_attribute_embeddings")
@patch.object(service, "_refresh_vdb_embeddings")
@patch.object(service, "patch_catalog_node")
@patch.object(service, "fetch_node_properties_by_id")
def test_patched_sample_values_come_back_as_text(
    mock_fetch: MagicMock,
    mock_patch: MagicMock,
    _mock_vdb: MagicMock,
    _mock_semantic: MagicMock,
) -> None:
    """The write reads back as stored JSON; the response is a string list."""
    mock_fetch.return_value = {"data_type": "text"}
    mock_patch.return_value = _patched_column(sample_values="[10, 1.5, true]")

    result = service.update_node_properties("col-1", {"sample_values": ["10"]})

    assert result == {"id": "col-1", "sample_values": ["10", "1.5", "True"]}


@patch.object(service, "patch_catalog_node")
@patch.object(service, "fetch_node_properties_by_id")
def test_sample_values_are_refused_on_a_non_text_column(
    mock_fetch: MagicMock,
    mock_patch: MagicMock,
) -> None:
    """The refusal must precede the write, or a rejected patch still lands."""
    mock_fetch.return_value = {"data_type": "integer"}

    with pytest.raises(ValueError, match="integer"):
        service.update_node_properties("col-1", {"sample_values": ["abc"]})

    mock_patch.assert_not_called()


@patch.object(service, "patch_catalog_node")
@patch.object(service, "fetch_node_properties_by_id")
def test_sample_values_are_refused_for_pii_or_unprocessed_column(
    mock_fetch: MagicMock,
    mock_patch: MagicMock,
    allow_sample_data_movement: MagicMock,
) -> None:
    mock_fetch.return_value = {"data_type": "text"}
    allow_sample_data_movement.return_value = False

    with pytest.raises(ValueError, match="PII classification"):
        service.update_node_properties("col-1", {"sample_values": ["secret"]})

    mock_patch.assert_not_called()


@patch.object(service, "fetch_parent_table_id_for_column", return_value="tbl-1")
@patch.object(service, "_refresh_vdb_embeddings")
@patch.object(service, "patch_catalog_node")
@patch.object(service, "fetch_node_properties_by_id")
def test_other_properties_of_a_non_text_column_stay_editable(
    mock_fetch: MagicMock,
    mock_patch: MagicMock,
    _mock_vdb: MagicMock,
    _mock_parent: MagicMock,
) -> None:
    """Only sample values are typed; a description is text whatever the column is."""
    mock_fetch.return_value = {"data_type": "integer"}
    mock_patch.return_value = _patched_column(description="Row count")

    result = service.update_node_properties("col-1", {"description": "Row count"})

    assert result == {"id": "col-1", "description": "Row count"}
    mock_fetch.assert_not_called()


@patch.object(service, "_refresh_semantic_column_attribute_embeddings")
@patch.object(service, "_refresh_vdb_embeddings")
@patch.object(service, "patch_catalog_node")
@patch.object(service, "fetch_node_properties_by_id")
def test_sample_values_on_a_node_that_is_no_column_are_left_alone(
    mock_fetch: MagicMock,
    mock_patch: MagicMock,
    _mock_vdb: MagicMock,
    _mock_semantic: MagicMock,
) -> None:
    """No Column, no declared type to check against — the write decides."""
    mock_fetch.return_value = None
    mock_patch.return_value = {
        "id": "tbl-1",
        "label": Labels.TABLE,
        "props": {"sample_values": ["x"]},
    }

    result = service.update_node_properties("tbl-1", {"sample_values": ["x"]})

    assert result == {"id": "tbl-1", "sample_values": ["x"]}
