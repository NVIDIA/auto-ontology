# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :func:`gsf.utils.embedding.batch_embed`.

``batch_embed`` exists to keep GSF's dependency on the private library symbol
``_BatchEmbedActor`` to one place. The test that matters most is therefore the
last one — that no other GSF module imports it.
"""

from __future__ import annotations

import ast
import pathlib

import pandas as pd

from gsf.utils.embedding import batch_embed

GSF_ROOT = pathlib.Path(__file__).resolve().parents[2]
WRAPPER = GSF_ROOT / "utils" / "embedding.py"


class _FakeParams:
    model_name = "fake"


def test_empty_rows_short_circuit_without_building_a_graph() -> None:
    """An empty ingest must not construct an embed graph or call the endpoint.

    ``gsf.utils.embedding`` imports ``Graph`` lazily inside the function, so a
    non-short-circuiting implementation would fail here on the missing endpoint
    rather than returning cleanly.
    """
    assert batch_embed([], _FakeParams()).empty
    assert batch_embed(pd.DataFrame(), _FakeParams()).empty


def test_private_actor_is_imported_in_exactly_one_place() -> None:
    """The whole point of the wrapper.

    ``_BatchEmbedActor`` is private to ``nemo_retriever``; every extra import of
    it is another site to fix when it moves. Keeping it to one file is the
    difference between a one-line change and a hunt.
    """
    allowed = {WRAPPER, pathlib.Path(__file__).resolve()}
    offenders = [
        str(path.relative_to(GSF_ROOT))
        for path in GSF_ROOT.rglob("*.py")
        if "_BatchEmbedActor" in path.read_text() and path.resolve() not in allowed
    ]
    assert offenders == [], (
        "import _BatchEmbedActor via gsf.utils.embedding.batch_embed instead: "
        f"{offenders}"
    )


def test_wrapper_confines_the_import_to_one_statement() -> None:
    """One import statement, so the blast radius is a single line."""
    tree = ast.parse(WRAPPER.read_text())
    imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and any(alias.name == "_BatchEmbedActor" for alias in node.names)
    ]
    assert len(imports) == 1
