# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :func:`auto_ontology.utils.embedding.batch_embed`.

``batch_embed`` exists to keep Auto Ontology's dependency on the private library symbol
``_BatchEmbedActor`` to one place. The test that matters most is therefore the
last one — that no other Auto Ontology module imports it.
"""

from __future__ import annotations

import ast
import logging
import pathlib

import pandas as pd
import pytest

from auto_ontology.utils.embedding import _embed_with_retry, batch_embed

AUTO_ONTOLOGY_ROOT = pathlib.Path(__file__).resolve().parents[2]
WRAPPER = AUTO_ONTOLOGY_ROOT / "utils" / "embedding.py"


class _FakeParams:
    model_name = "fake"


def test_empty_rows_short_circuit_without_building_a_graph() -> None:
    """An empty ingest must not construct an embed graph or call the endpoint.

    ``auto_ontology.utils.embedding`` imports ``Graph`` lazily inside the function, so a
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
        str(path.relative_to(AUTO_ONTOLOGY_ROOT))
        for path in AUTO_ONTOLOGY_ROOT.rglob("*.py")
        if "_BatchEmbedActor" in path.read_text() and path.resolve() not in allowed
    ]
    assert offenders == [], (
        "import _BatchEmbedActor via auto_ontology.utils.embedding.batch_embed instead: "
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


# ---------------------------------------------------------------------------
# Retry
# ---------------------------------------------------------------------------
#
# The endpoint rate-limits: a sustained ingest gets 429 once its quota is gone.
# The library turns that into rows with empty embeddings rather than an
# exception, so without a retry those rows are silently dropped and the catalog
# is quietly incomplete. These tests pin the behaviour that prevents that.

_RATE_LIMITED = "Embedding error occurred: 429 Too Many Requests"


def _embedded(rows: int = 2) -> pd.DataFrame:
    return pd.DataFrame({"metadata": [{"embedding": [0.1, 0.2]}] * rows})


def _refused(error: str = _RATE_LIMITED) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metadata": [{"embedding": []}] * 2,
            "text_embeddings_1b_v2": [{"embedding": [], "error": error}] * 2,
        }
    )


def _part() -> pd.DataFrame:
    return pd.DataFrame({"text": ["a", "b"]})


def test_retry_recovers_a_rate_limited_chunk() -> None:
    """Two 429s then success must yield the embeddings, not an empty frame."""
    attempts = [_refused(), _refused(), _embedded()]
    slept: list[float] = []

    result = _embed_with_retry(
        lambda part: attempts.pop(0),
        _part(),
        label="chunk 1/1",
        sleep=slept.append,
    )

    assert result is not None and not result.empty
    assert attempts == []
    # Backoff must grow: an immediate retry just hits the same closed window.
    assert slept == [15.0, 30.0]


def test_retry_gives_up_and_reports_the_endpoint_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """After the last attempt the endpoint's own error must reach the log."""
    calls: list[int] = []

    def _always_refused(part: pd.DataFrame) -> pd.DataFrame:
        calls.append(1)
        return _refused()

    with caplog.at_level(logging.WARNING):
        result = _embed_with_retry(
            _always_refused,
            _part(),
            label="chunk 1/1",
            attempts=3,
            sleep=lambda _seconds: None,
        )

    assert len(calls) == 3
    assert result is not None
    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "giving up after 3 attempts" in text
    assert "429 Too Many Requests" in text


def test_first_attempt_success_does_not_sleep() -> None:
    """The happy path must not pay any backoff."""
    slept: list[float] = []

    result = _embed_with_retry(
        lambda part: _embedded(),
        _part(),
        label="chunk 1/1",
        sleep=slept.append,
    )

    assert result is not None and not result.empty
    assert slept == []
