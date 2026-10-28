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
import threading
from typing import Any

import pytest

from auto_ontology.ingestion_service.data_scheduler import DataScheduler
from auto_ontology.ingestion_service.semantic_scheduler import SemanticScheduler


class _Connector:
    def __init__(self, name: str) -> None:
        self.database_name = name


def _messages(caplog: pytest.LogCaptureFixture) -> str:
    return "\n".join(record.getMessage() for record in caplog.records)


def _stub_rules(monkeypatch: pytest.MonkeyPatch, rules: Any = None) -> list[bool]:
    """Neutralise the rule replay, and report whether it ran.

    Stubbed like the compilation itself: it reads the rules out of Postgres,
    which these tests have no database for, and it is each caller's decision
    to replay them that is under test here rather than the rules' own
    behaviour. The returned list has an entry per call, so "the rules ran" and
    "the rules did not" are both checkable.
    """
    import auto_ontology.ingestion_service.rules as mod

    ran: list[bool] = []

    def _default() -> tuple[int, int]:
        ran.append(True)
        return 0, 0

    monkeypatch.setattr(mod, "reapply_rules", rules or _default)
    return ran


# --------------------------------------------------------------------------
# run_ingest
# --------------------------------------------------------------------------


@pytest.fixture
def _stub_ingest(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralise everything run_ingest touches except the logging."""
    import auto_ontology.ingestion_service.ingest as mod

    monkeypatch.setattr(mod, "get_embed_params", lambda: {})
    monkeypatch.setattr(mod, "ingest_catalog", lambda c: ([1, 2, 3], [1, 2]))
    monkeypatch.setattr(mod, "is_pii_detection_enabled", lambda: True)
    monkeypatch.setattr(mod, "detect_and_tag_pii", lambda columns: None)
    monkeypatch.setattr(mod, "run_pii_propagation", lambda label: None)
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


def test_run_ingest_tags_persisted_columns_before_embedding(
    _stub_ingest: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    import auto_ontology.ingestion_service.ingest as mod

    order: list[str] = []

    def ingest(_connector: Any) -> tuple[list[int], list[int]]:
        order.append("catalog")
        return [1], [2]

    def detect(_columns: Any) -> None:
        order.append("pii")

    def embedding_rows(**_kwargs: Any) -> Any:
        def build(pair: Any) -> Any:
            order.append("embedding")
            return pair

        return build

    def propagate(_label: str) -> None:
        order.append("propagate")

    monkeypatch.setattr(mod, "ingest_catalog", ingest)
    monkeypatch.setattr(mod, "detect_and_tag_pii", detect)
    monkeypatch.setattr(mod, "run_pii_propagation", propagate)
    monkeypatch.setattr(mod, "CatalogEmbeddingRowsOp", embedding_rows)

    mod.run_ingest(_Connector("pagila"))

    assert order == ["catalog", "pii", "propagate", "embedding"]


def test_detection_failure_still_propagates_pii_to_attributes(
    _stub_ingest: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    import auto_ontology.ingestion_service.ingest as mod

    propagated: list[str] = []

    def fail(_columns: Any) -> None:
        raise RuntimeError("classifier unavailable")

    monkeypatch.setattr(mod, "detect_and_tag_pii", fail)
    monkeypatch.setattr(mod, "run_pii_propagation", propagated.append)

    mod.run_ingest(_Connector("pagila"))

    assert propagated == ["pagila"]


def test_propagation_failure_is_contained(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import auto_ontology.ingestion_service.pii as pii_mod

    def boom() -> None:
        raise RuntimeError("db down")

    monkeypatch.setattr(pii_mod, "propagate_pii_to_attributes", boom)

    with caplog.at_level(logging.INFO):
        pii_mod.run_pii_propagation("ingest")

    assert "could not propagate PII tags to attributes" in _messages(caplog)


def test_run_ingest_skips_pii_when_disabled(
    _stub_ingest: None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import auto_ontology.ingestion_service.ingest as mod

    pii_runs: list[str] = []
    monkeypatch.setattr(mod, "is_pii_detection_enabled", lambda: False)
    monkeypatch.setattr(
        mod, "detect_and_tag_pii", lambda _columns: pii_runs.append("detect")
    )
    monkeypatch.setattr(
        mod, "run_pii_propagation", lambda _label: pii_runs.append("propagate")
    )

    with caplog.at_level(logging.INFO):
        mod.run_ingest(_Connector("pagila"))

    assert pii_runs == []
    assert "PII detection skipped for database pagila" in _messages(caplog)
    assert "Data ingestion finished successfully for database pagila" in _messages(
        caplog
    )


def test_pii_failure_does_not_fail_catalog_ingestion(
    _stub_ingest: None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import auto_ontology.ingestion_service.ingest as mod

    def fail(_columns: Any) -> None:
        raise RuntimeError("classifier unavailable")

    monkeypatch.setattr(mod, "detect_and_tag_pii", fail)

    with caplog.at_level(logging.INFO):
        mod.run_ingest(_Connector("pagila"))

    text = _messages(caplog)
    assert "PII detection failed for database pagila" in text
    assert "Unprocessed columns will be retried on the next ingest" in text
    assert "Data ingestion finished successfully for database pagila" in text


def test_trigger_ingest_re_applies_the_rules(
    _stub_ingest: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Neither scheduler covers this path, so it has to replay them itself.

    Creating a connection asks the ingestion service for an ingest and not for
    a compilation, so without this the database that was just ingested carries
    no rule's tags until a scheduler tick hours later.
    """
    import auto_ontology.ingestion_service.ingest as mod

    monkeypatch.setattr(mod, "invalidate_connectors_cache", lambda: None)
    monkeypatch.setattr(mod, "get_connectors", lambda: [_Connector("pagila")])
    pii_runs: list[bool] = []
    monkeypatch.setattr(
        mod, "detect_and_tag_pii", lambda _columns: pii_runs.append(True)
    )
    ran = _stub_rules(monkeypatch)

    mod.trigger_ingest({"database": "pagila"})
    for thread in threading.enumerate():
        if thread.name == "ingest-pagila":
            thread.join(timeout=10)

    assert ran == [True]
    assert pii_runs == [True]


# --------------------------------------------------------------------------
# DataScheduler
# --------------------------------------------------------------------------


def _patch_data_scheduler(
    monkeypatch: pytest.MonkeyPatch,
    connectors: list[_Connector],
    run: Any,
    *,
    semantic_enabled: bool = False,
) -> list[bool]:
    import auto_ontology.ingestion_service.data_scheduler as mod

    monkeypatch.setattr(mod, "invalidate_connectors_cache", lambda: None)
    monkeypatch.setattr(mod, "get_connectors", lambda: connectors)
    monkeypatch.setattr(mod, "run_ingest", run)
    monkeypatch.setattr(
        mod, "is_semantic_compilation_enabled", lambda: semantic_enabled
    )
    return _stub_rules(monkeypatch)


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
# Tagging rules, at the end of an ingest pass
# --------------------------------------------------------------------------
#
# Only when semantic compilation is off. With it on, the semantic pass replays
# the rules against both layers at once, which is the better moment; with it
# off there is no such pass, and without this the deployment would label the
# catalog once per rule, when the rule was created, and never again.


def test_ingest_pass_re_applies_the_rules_when_compilation_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ran = _patch_data_scheduler(
        monkeypatch, [_Connector("a")], lambda c: None, semantic_enabled=False
    )

    asyncio.run(DataScheduler()._run_once())

    assert ran == [True]


def test_ingest_pass_leaves_the_rules_to_the_semantic_pass_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ran = _patch_data_scheduler(
        monkeypatch, [_Connector("a")], lambda c: None, semantic_enabled=True
    )

    asyncio.run(DataScheduler()._run_once())

    assert ran == []


def test_a_failure_to_re_apply_does_not_fail_the_ingest(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def _boom() -> tuple[int, int]:
        raise RuntimeError("no database")

    _patch_data_scheduler(monkeypatch, [_Connector("a")], lambda c: None)
    _stub_rules(monkeypatch, _boom)

    with caplog.at_level(logging.INFO):
        asyncio.run(DataScheduler()._run_once())

    text = _messages(caplog)
    assert "could not re-apply tagging rules" in text
    assert "ingest: finished successfully — 1 connection(s)" in text


# --------------------------------------------------------------------------
# SemanticScheduler
# --------------------------------------------------------------------------


#: Filled by the stubbed PII propagation; reset by each ``_patch_semantic_scheduler``.
_PROPAGATED: list[bool] = []


def _patch_semantic_scheduler(
    monkeypatch: pytest.MonkeyPatch,
    databases: list[str],
    run: Any,
    rules: Any = None,
) -> list[bool]:
    """Neutralise the pass's dependencies, and report whether rules ran."""
    import auto_ontology.ingestion_service.semantic_scheduler as mod

    monkeypatch.setattr(mod, "is_semantic_compilation_enabled", lambda: True)
    monkeypatch.setattr(mod, "resolve_database_names", lambda: databases)
    monkeypatch.setattr(mod, "run_semantic_compilation", run)

    async def _record_propagation(_label: str) -> None:
        _PROPAGATED.append(True)

    _PROPAGATED.clear()
    monkeypatch.setattr(mod, "run_pii_propagation_in_thread", _record_propagation)
    return _stub_rules(monkeypatch, rules)


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


# --------------------------------------------------------------------------
# Tagging rules, at the end of a semantic pass
# --------------------------------------------------------------------------
#
# A rule labels both layers, so this pass is the only point at which the
# catalog and the semantic layer it labels are both current. What the tests
# below pin is *when* it runs: after a completed pass, not after one that was
# cut short, since a stop leaves the semantic layer half written and labels
# applied against half of it would be taken back on the next pass.


def test_a_completed_pass_re_applies_the_rules(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    ran = _patch_semantic_scheduler(monkeypatch, ["pagila"], lambda db: 7)

    with caplog.at_level(logging.INFO):
        asyncio.run(SemanticScheduler()._run_once())

    assert ran == [True]


def test_rules_still_run_when_one_database_failed_to_compile(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The pass ran to the end; what it managed to write is worth labelling."""

    def _sometimes(database_name: str) -> int:
        if database_name == "chinook":
            raise RuntimeError("nope")
        return 7

    ran = _patch_semantic_scheduler(monkeypatch, ["pagila", "chinook"], _sometimes)

    with caplog.at_level(logging.INFO):
        asyncio.run(SemanticScheduler()._run_once())

    assert ran == [True]


def test_a_stopped_pass_does_not_re_apply_the_rules(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Stopped at a database boundary, so the semantic layer is half written."""
    ran = _patch_semantic_scheduler(monkeypatch, ["pagila", "chinook"], lambda db: 7)
    scheduler = SemanticScheduler()
    scheduler.abort()

    with caplog.at_level(logging.INFO):
        asyncio.run(scheduler._run_once())

    assert ran == []


def test_a_disabled_pass_does_not_re_apply_the_rules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ran = _patch_semantic_scheduler(monkeypatch, ["pagila"], lambda db: 7)
    import auto_ontology.ingestion_service.semantic_scheduler as mod

    monkeypatch.setattr(mod, "is_semantic_compilation_enabled", lambda: False)

    asyncio.run(SemanticScheduler()._run_once())

    assert ran == []


def test_a_completed_pass_propagates_pii_to_the_new_attributes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Attributes are created by compilation, after the ingest that tagged
    their columns, so this pass has to carry the PII tag onto them."""
    _patch_semantic_scheduler(monkeypatch, ["pagila"], lambda db: 7)

    asyncio.run(SemanticScheduler()._run_once())

    assert _PROPAGATED == [True]


def test_a_stopped_pass_does_not_propagate_pii(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_semantic_scheduler(monkeypatch, ["pagila"], lambda db: 7)
    scheduler = SemanticScheduler()
    scheduler.abort()

    asyncio.run(scheduler._run_once())

    assert _PROPAGATED == []


def test_a_pass_stopped_during_its_last_database_skips_post_compilation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = SemanticScheduler()

    def stop_during_compile(_database_name: str) -> int:
        scheduler.abort()
        return 7

    rules = _patch_semantic_scheduler(monkeypatch, ["pagila"], stop_during_compile)

    asyncio.run(scheduler._run_once())

    assert _PROPAGATED == []
    assert rules == []


def test_a_failure_to_re_apply_does_not_fail_the_compilation(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Otherwise a rules outage would put a red state on the settings page
    against a compilation that in fact succeeded."""

    def _boom() -> tuple[int, int]:
        raise RuntimeError("no database")

    _patch_semantic_scheduler(monkeypatch, ["pagila"], lambda db: 7, rules=_boom)

    with caplog.at_level(logging.INFO):
        asyncio.run(SemanticScheduler()._run_once())

    text = _messages(caplog)
    assert "could not re-apply tagging rules" in text
    assert "semantic: finished successfully — 1 database(s)" in text
