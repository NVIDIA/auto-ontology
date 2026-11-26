# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Search stores are built once per collection, not once per query."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import pytest

import auto_ontology.vdb as vdb


@pytest.fixture
def built() -> Iterator[list[str]]:
    collections: list[str] = []

    def _build(**kwargs: Any) -> object:
        collections.append(kwargs["collection_name"])
        return object()

    vdb.shared_vdb.cache_clear()
    with patch.object(vdb, "get_vdb", side_effect=_build):
        yield collections
    vdb.shared_vdb.cache_clear()


def test_repeated_searches_share_one_store(built: list[str]) -> None:
    first = vdb.shared_vdb(vdb.CONVERSATION_VDB_COLLECTION)
    again = vdb.shared_vdb(vdb.CONVERSATION_VDB_COLLECTION)

    assert first is again
    assert built == [vdb.CONVERSATION_VDB_COLLECTION]


def test_each_collection_has_its_own_store(built: list[str]) -> None:
    nodes = vdb.shared_vdb(vdb.CONVERSATION_SEMANTIC_VDB_COLLECTION)
    paths = vdb.shared_vdb(vdb.ONE_HOP_PATH_VDB_COLLECTION)

    assert nodes is not paths
    assert len(built) == 2
