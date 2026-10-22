# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from auto_ontology.semantic import compile as semantic_compile
from auto_ontology.semantic import sql_attribute_suggester


def test_role_enrichment_runs_after_semantic_fk_resolution(monkeypatch) -> None:
    calls: list[str] = []
    embedder = object()
    monkeypatch.setattr(
        semantic_compile,
        "build_semantic_embedder",
        lambda *_args, **_kwargs: embedder,
    )
    monkeypatch.setattr(
        semantic_compile,
        "compile_semantic_layer",
        lambda *_args, **_kwargs: calls.append("tables") or 2,
    )
    monkeypatch.setattr(
        semantic_compile,
        "resolve_semantic_fks",
        lambda *_args, **_kwargs: calls.append("semantic_fks") or 3,
    )
    monkeypatch.setattr(
        semantic_compile,
        "enrich_semantic_fk_roles",
        lambda *_args, **kwargs: (
            calls.append(f"roles:{kwargs['embedder'] is embedder}") or 3
        ),
    )
    monkeypatch.setattr(
        sql_attribute_suggester,
        "suggest_sql_attributes",
        lambda *_args, **_kwargs: calls.append("sql_attributes") or 0,
    )
    monkeypatch.setattr(
        semantic_compile,
        "build_bridge_tables_sql_attributes",
        lambda *_args, **_kwargs: calls.append("bridges") or 0,
    )

    result = semantic_compile._run_semantic_compilation(
        "database",
        domain_summary=object(),
    )

    assert result == 2
    assert calls == [
        "tables",
        "semantic_fks",
        "roles:True",
        "sql_attributes",
        "bridges",
    ]
