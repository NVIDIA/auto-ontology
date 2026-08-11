# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pin the forked catalog vocabulary to the library it was forked from.

:mod:`gsf.catalog.constants` is a verbatim fork of
``nemo_retriever.tabular_data.ingestion.model.reserved_words``. Both still write
and read the same graph, so a divergence — a value the library changed, or a
label it added that GSF's writers would then not know about — would corrupt the
catalog silently.

**Delete this test in Phase 11**, once nothing in GSF imports the library's
ingestion package. See ``docs/refactor/drop-neo4j/PLAN.md``.
"""

from __future__ import annotations

from typing import Any

import pytest
from nemo_retriever.tabular_data.ingestion.model import reserved_words as library

from gsf.catalog import constants as fork

VOCABULARY = ("Labels", "TableTypes", "Edges", "Props")


def _public_attrs(namespace: type) -> dict[str, Any]:
    """Class attributes that make up the vocabulary, docstrings excluded."""
    return {
        name: value
        for name, value in vars(namespace).items()
        if not name.startswith("_")
    }


@pytest.mark.parametrize("name", VOCABULARY)
def test_fork_matches_library(name: str) -> None:
    assert _public_attrs(getattr(fork, name)) == _public_attrs(
        getattr(library, name)
    ), f"{name} has diverged from the library it was forked from"


def test_no_vocabulary_class_was_missed() -> None:
    """Every class the library exports is one GSF also owns."""
    exported = {
        name
        for name, value in vars(library).items()
        if isinstance(value, type) and not name.startswith("_")
    }
    assert exported == set(VOCABULARY)
