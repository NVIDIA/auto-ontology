# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Restoring a declared primary key that the vector index does not carry.

The index stores a hit's text and a handful of metadata fields, none of which is
the key, so a table assembled from a vector hit arrives without one however
clearly the source declared it. Nothing downstream can recover it: an edge is
oriented towards a primary key and a prediction names its entity by one.

The shapes here are the ones the evaluation sweep failed on, taken from the
sqlite files themselves: ``Store Locations`` keyed on ``StoreID``, and
``product`` keyed on ``("Product ID", "Region")``.
"""

from unittest.mock import patch

import pytest

from gsf.retrieval.data_access import relevant_tables as module
from gsf.retrieval.data_access.relevant_tables import restore_primary_keys


@pytest.fixture()
def catalog():
    with patch.object(module, "fetch_primary_keys_by_ids") as fetch:
        yield fetch


def test_a_declared_key_is_restored(catalog) -> None:
    """Held as a list, the shape ``pk`` carries everywhere else in the catalog."""
    catalog.return_value = {"t1": ["StoreID"]}

    tables = restore_primary_keys([{"id": "t1", "name": "Store Locations"}])

    assert tables[0]["primary_key"] == ["StoreID"]


def test_a_composite_key_is_restored_whole(catalog) -> None:
    """``product`` is keyed on two columns; ``Product ID`` alone repeats."""
    catalog.return_value = {"t2": ["Product ID", "Region"]}

    tables = restore_primary_keys([{"id": "t2", "name": "product"}])

    assert tables[0]["primary_key"] == ["Product ID", "Region"]


def test_a_key_already_present_is_left_alone(catalog) -> None:
    """A caller that supplied one holds the richer answer."""
    tables = restore_primary_keys(
        [{"id": "t1", "name": "Store Locations", "primary_key": "StoreID"}]
    )

    catalog.assert_not_called()
    assert tables[0]["primary_key"] == "StoreID"


def test_a_table_the_catalog_has_no_key_for_stays_without_one(catalog) -> None:
    catalog.return_value = {}

    tables = restore_primary_keys([{"id": "t9", "name": "notes"}])

    assert "primary_key" not in tables[0]


def test_only_the_tables_missing_a_key_are_asked_about(catalog) -> None:
    catalog.return_value = {"t2": ["Product ID", "Region"]}

    restore_primary_keys(
        [
            {"id": "t1", "name": "Store Locations", "primary_key": "StoreID"},
            {"id": "t2", "name": "product"},
        ]
    )

    catalog.assert_called_once_with(["t2"])


def test_a_catalog_failure_leaves_the_tables_untouched(catalog) -> None:
    """The backfill fills in what is missing; failing to fill is not a failure."""
    catalog.return_value = {}

    tables = restore_primary_keys([{"id": "t1", "name": "Store Locations"}])

    assert tables == [{"id": "t1", "name": "Store Locations"}]


def test_a_table_without_an_id_is_skipped(catalog) -> None:
    catalog.return_value = {}

    tables = restore_primary_keys([{"name": "orphan"}])

    catalog.assert_not_called()
    assert tables == [{"name": "orphan"}]
