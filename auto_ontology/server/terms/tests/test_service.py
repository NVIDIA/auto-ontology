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


@patch(
    "auto_ontology.server.terms.service.is_column_id_safe_for_data_movement",
    return_value=False,
)
@patch("auto_ontology.dal.attributes.update_column_attribute")
@patch("auto_ontology.dal.attributes.fetch_column_attribute_column")
def test_pii_sample_values_are_refused_before_metadata_write(
    mock_fetch_column: MagicMock,
    mock_update: MagicMock,
    _mock_safe: MagicMock,
) -> None:
    mock_fetch_column.return_value = {"id": "col-1", "data_type": "text"}

    with pytest.raises(ValueError, match="classified as non-PII"):
        service.update_column_attribute(
            "term-1",
            "attr-1",
            name="Renamed",
            sample_values=["secret"],
        )

    mock_update.assert_not_called()
