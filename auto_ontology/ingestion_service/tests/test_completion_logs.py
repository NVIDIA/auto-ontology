# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The "finished successfully" lines must mean it, and only when it did.

These logs exist to answer one operational question from a pod log: did the
ingest / semantic pass actually complete? That is only worth anything if the
line cannot appear after a failure. Both schedulers deliberately swallow
per-item exceptions so one bad connection does not stop the rest, which is
exactly what made the previous bare ``ingest: finished`` useless — it printed
whether every connection ingested or every one of them threw.

So each pipeline is tested twice: once succeeding, once failing.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from auto_ontology.ingestion_service.data_scheduler import DataScheduler
from auto_ontology.ingestion_service.semantic_scheduler import SemanticScheduler


class _Connector:
    def __init__(self, name: str) -> None:
        self.database_name = name


def _messages(caplog: pytest.LogCaptureFixture) -> str:
    return "\n".join(record.getMessage() for record in caplog.records)


# --------------------------------------------------------------------------
# run_ingest
# --------------------------------------------------------------------------


@pytest.fixture
def _stub_ingest(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralise everything run_ingest touches except the logging."""
    import auto_ontology.ingestion_service.ingest as mod

    monkeypatch.setattr(mod, "get_embed_params", lambda: {})
    monkeypatch.setattr(mod, "ingest_catalog", lambda c: ([1, 2, 3], [1, 2]))
    monkeypatch.setattr(mod, "CatalogEmbeddingRowsOp", lambda **kw: lambda pair: pair)
    monkeypatch.setattr(
        mod, "batch_embed_chunks", lambda rows, params, label="": iter(())
    )


def test_run_ingest_logs_success_with_counts(
    _stub_ingest: None, caplog: pytest.LogCaptureFixture
) -> None:
    from auto_ontology.ingestion_service.ingest import run_ingest

    with caplog.at_level(logging.INFO):
        run_ingest(_Connector("pagila"))

    text = _messages(caplog)
    assert "Data ingestion started for database pagila" in text
    assert "Data ingestion finished successfully for database pagila" in text
    # The counts come from the catalog frames, not from the embed step.
    assert "3 table(s), 2 column(s)" in text


def test_run_ingest_logs_the_empty_embed_case(
    _stub_ingest: None, caplog: pytest.LogCaptureFixture
) -> None:
    """An empty embed result used to log nothing at all."""
    from auto_ontology.ingestion_service.ingest import run_ingest

    with caplog.at_level(logging.INFO):
        run_ingest(_Connector("pagila"))

    assert "nothing written to pgvector" in _messages(caplog)


def test_run_ingest_logs_no_success_when_extraction_raises(
    _stub_ingest: None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import auto_ontology.ingestion_service.ingest as mod

    def _boom(connector: Any) -> Any:
        raise RuntimeError("catalog extraction failed")

    monkeypatch.setattr(mod, "ingest_catalog", _boom)

    with caplog.at_level(logging.INFO), pytest.raises(RuntimeError):
        mod.run_ingest(_Connector("pagila"))

    assert "finished successfully" not in _messages(caplog)


# --------------------------------------------------------------------------
# DataScheduler
# --------------------------------------------------------------------------


def _patch_data_scheduler(
    monkeypatch: pytest.MonkeyPatch, connectors: list[_Connector], run: Any
) -> None:
    import auto_ontology.ingestion_service.data_scheduler as mod

    monkeypatch.setattr(mod, "invalidate_connectors_cache", lambda: None)
    monkeypatch.setattr(mod, "get_connectors", lambda: connectors)
    monkeypatch.setattr(mod, "run_ingest", run)


def test_data_scheduler_reports_success_tally(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _patch_data_scheduler(
        monkeypatch, [_Connector("a"), _Connector("b")], lambda c: None
    )

    with caplog.at_level(logging.INFO):
        asyncio.run(DataScheduler()._run_once())

    assert "ingest: finished successfully — 2 connection(s)" in _messages(caplog)


def test_data_scheduler_does_not_claim_success_when_all_fail(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The regression the tally was added for."""

    def _boom(connector: Any) -> None:
        raise RuntimeError("nope")

    _patch_data_scheduler(monkeypatch, [_Connector("a"), _Connector("b")], _boom)

    with caplog.at_level(logging.INFO):
        asyncio.run(DataScheduler()._run_once())

    text = _messages(caplog)
    assert "finished successfully" not in text
    assert "finished with errors — 0 of 2 connection(s) succeeded, 2 failed" in text


def test_data_scheduler_partial_failure_is_reported_as_such(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def _sometimes(connector: Any) -> None:
        if connector.database_name == "b":
            raise RuntimeError("nope")

    _patch_data_scheduler(monkeypatch, [_Connector("a"), _Connector("b")], _sometimes)

    with caplog.at_level(logging.INFO):
        asyncio.run(DataScheduler()._run_once())

    text = _messages(caplog)
    assert "finished successfully" not in text
    assert "1 of 2 connection(s) succeeded, 1 failed" in text


# --------------------------------------------------------------------------
# SemanticScheduler
# --------------------------------------------------------------------------


def _patch_semantic_scheduler(
    monkeypatch: pytest.MonkeyPatch, databases: list[str], run: Any
) -> None:
    import auto_ontology.ingestion_service.semantic_scheduler as mod

    monkeypatch.setattr(mod, "is_semantic_compilation_enabled", lambda: True)
    monkeypatch.setattr(mod, "resolve_database_names", lambda: databases)
    monkeypatch.setattr(mod, "run_semantic_compilation", run)


def test_semantic_scheduler_reports_success_and_table_total(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _patch_semantic_scheduler(monkeypatch, ["pagila", "chinook"], lambda db: 7)

    with caplog.at_level(logging.INFO):
        asyncio.run(SemanticScheduler()._run_once())

    text = _messages(caplog)
    assert "Finished semantic compilation successfully for database pagila" in text
    # 7 per database, summed across both.
    assert "semantic: finished successfully — 2 database(s), 14 table(s)" in text
    # The "nothing to compile" note belongs only on a pass that compiled nothing.
    assert "no table needed compiling" not in text


def test_semantic_scheduler_explains_a_zero_table_pass(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A settled catalog compiles nothing, but the pass is not therefore idle.

    The count is tables that *needed* compiling, while the elapsed time also
    covers the FK / SqlAttribute / bridge-table stages, which run every pass. A
    bare "0 table(s) ... in 620.7s" reads as a stall or a failed run, so the
    zero case has to say why.
    """
    _patch_semantic_scheduler(monkeypatch, ["pagila", "chinook"], lambda db: 0)

    with caplog.at_level(logging.INFO):
        asyncio.run(SemanticScheduler()._run_once())

    text = _messages(caplog)
    assert (
        "semantic: finished successfully — 2 database(s), 0 table(s) compiled" in text
    )
    assert "no table needed compiling" in text
    assert (
        "Finished semantic compilation successfully for database pagila: "
        "0 table(s) compiled" in text
    )
    # The explanation belongs on the summary only -- once, not once per database.
    assert text.count("no table needed compiling") == 1


def test_semantic_scheduler_does_not_claim_success_when_one_fails(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def _sometimes(database_name: str) -> int:
        if database_name == "chinook":
            raise RuntimeError("nope")
        return 7

    _patch_semantic_scheduler(monkeypatch, ["pagila", "chinook"], _sometimes)

    with caplog.at_level(logging.INFO):
        asyncio.run(SemanticScheduler()._run_once())

    text = _messages(caplog)
    assert "semantic: finished successfully" not in text
    assert "1 of 2 database(s) succeeded, 1 failed" in text


def test_semantic_scheduler_disabled_run_claims_nothing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A disabled pass returns early and must not log a completion."""
    import auto_ontology.ingestion_service.semantic_scheduler as mod

    monkeypatch.setattr(mod, "is_semantic_compilation_enabled", lambda: False)

    with caplog.at_level(logging.INFO):
        asyncio.run(SemanticScheduler()._run_once())

    assert "finished successfully" not in _messages(caplog)
