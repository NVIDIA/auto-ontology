# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pin the forked catalog write path to the library it was forked from.

Phase 1 forked ~2,500 lines out of ``nemo_retriever.tabular_data.ingestion``
**verbatim** — Phase 4 rewrites ``gsf/catalog/store/`` for Postgres, and
everything above it is meant to fork once and never again. Until Phase 11 both
copies still write the same Neo4j graph, so this suite guards two things:

* **no accidental edit** slipped into the fork while it was being moved, and
* **no upstream drift** landed in the library that GSF's copy would then miss.

Comparison is on the **AST**, not the text: the fork is ``ruff format``-ed to
this repo's style and carries a provenance docstring, neither of which is a
behaviour change. Module docstrings are dropped from both sides before
comparing.

Phase 11 took the narrower-list option this docstring offered. ``gsf/catalog/store/``
is no longer a fork of anything — it is the Postgres implementation — so every
mapping naming it is gone. What remains is the genuinely storage-agnostic part
of the fork: normalisation, SQL parsing, the model classes and the parsers.
Those still track upstream, and upstream drift in them is still worth catching.
"""

from __future__ import annotations

import ast
import importlib.util
import pathlib
import re

import pytest

# forked module -> library module it was forked from.
FORKED = {
    "gsf.catalog.normalize": "nemo_retriever.tabular_data.ingestion.utils",
    "gsf.catalog.sql_parse": "nemo_retriever.tabular_data.ingestion.services.queries",
    "gsf.catalog.model.node": "nemo_retriever.tabular_data.ingestion.model.neo4j_node",
    "gsf.catalog.model.schema": "nemo_retriever.tabular_data.ingestion.model.schema",
    "gsf.catalog.model.query": "nemo_retriever.tabular_data.ingestion.model.query",
    "gsf.catalog.parsers.sqlglot_extractor": (
        "nemo_retriever.tabular_data.ingestion.parsers.sqlglot_extractor"
    ),
    "gsf.catalog.parsers.query_comparator": (
        "nemo_retriever.tabular_data.ingestion.parsers.query_comparator"
    ),
    "gsf.catalog.parsers.schemas_parser": (
        "nemo_retriever.tabular_data.ingestion.parsers.schemas_parser"
    ),
    "gsf.catalog.services.schema": "nemo_retriever.tabular_data.ingestion.services.schema",
}

# Deliberately **not** verbatim, each with a reason. Anything added here needs a
# DECISIONS.md record first.
DIVERGED = {
    "gsf.catalog.extract": (
        "takes a connector instead of a library TabularExtractParams, and drops "
        "store_relational_db_in_neo4j (a two-line forwarder ingest_catalog now "
        "calls directly) — DECISION-004"
    ),
    "gsf.catalog.ingest": "new in Phase 1; replaces the library's TabularSchemaExtractOp",
    "gsf.catalog.query_stats": (
        "Phase 4: get_sql_counters and get_candidate_sql_ids are pure, so they "
        "moved out of store/neo4j/queries.py to be shared -- DECISION-007"
    ),
    "gsf.catalog.diff": (
        "Phase 4: the storage-agnostic re-ingest diff, lifted out of "
        "store/neo4j/db.py so both backends share one copy -- DECISION-006"
    ),
    "gsf.catalog.write": (
        "executor.map result is consumed, so a failing schema update raises "
        "instead of being discarded with its future — DECISION-006"
    ),
    "gsf.catalog.constants": "forked in Phase 0; covered by test_constants.py",
    # Phase 4 wrote these for Postgres and Phase 11 promoted them out of
    # `store/pg/`. Not forks of anything, and never were after Phase 4.
    "gsf.catalog.store.db": "Postgres implementation; not a fork",
    "gsf.catalog.store.schemas": "Postgres implementation; not a fork",
    "gsf.catalog.store.queries": "Postgres implementation; not a fork",
    "gsf.catalog.store.edges": "Postgres implementation; not a fork",
    "gsf.catalog.store.indexes": "Postgres implementation; not a fork",
    "gsf.catalog.store.registry": "Postgres implementation; not a fork",
    "gsf.catalog.store.rows": "Postgres implementation; not a fork",
}

MODULE_MAP = {
    "nemo_retriever.tabular_data.ingestion.model.reserved_words": "gsf.catalog.constants",
    "nemo_retriever.tabular_data.ingestion.model.neo4j_node": "gsf.catalog.model.node",
    "nemo_retriever.tabular_data.ingestion.model.schema": "gsf.catalog.model.schema",
    "nemo_retriever.tabular_data.ingestion.model.query": "gsf.catalog.model.query",
    "nemo_retriever.tabular_data.ingestion.parsers.sqlglot_extractor": (
        "gsf.catalog.parsers.sqlglot_extractor"
    ),
    "nemo_retriever.tabular_data.ingestion.parsers.query_comparator": (
        "gsf.catalog.parsers.query_comparator"
    ),
    "nemo_retriever.tabular_data.ingestion.parsers.schemas_parser": (
        "gsf.catalog.parsers.schemas_parser"
    ),
    "nemo_retriever.tabular_data.ingestion.parsers": "gsf.catalog.parsers",
    "nemo_retriever.tabular_data.ingestion.services.schema": "gsf.catalog.services.schema",
    "nemo_retriever.tabular_data.ingestion.services.queries": "gsf.catalog.sql_parse",
    # Import-path equivalences. These are not claims that the modules match —
    # `gsf.catalog.store.*` is the Postgres implementation and appears in
    # DIVERGED — only that an upstream module importing `queries_dal` and a
    # forked module importing `gsf.catalog.store.queries` are importing the
    # same thing by different names. Without them a *genuinely* verbatim fork
    # reads as drift on its import lines alone.
    "nemo_retriever.tabular_data.ingestion.dal.db_dal": "gsf.catalog.store.db",
    "nemo_retriever.tabular_data.ingestion.dal.schemas_dal": "gsf.catalog.store.schemas",
    "nemo_retriever.tabular_data.ingestion.dal.queries_dal": "gsf.catalog.store.queries",
    "nemo_retriever.tabular_data.ingestion.dal.utils_dal": "gsf.catalog.store.edges",
    "nemo_retriever.tabular_data.ingestion.indexes": "gsf.catalog.store.indexes",
    "nemo_retriever.tabular_data.ingestion.utils": "gsf.catalog.normalize",
    "nemo_retriever.tabular_data.ingestion.extract_data": "gsf.catalog.extract",
    "nemo_retriever.tabular_data.ingestion.write_to_graph": "gsf.catalog.write",
}

CATALOG_DIR = pathlib.Path(__file__).resolve().parents[1]


def _source(module: str) -> str:
    spec = importlib.util.find_spec(module)
    assert spec is not None and spec.origin, f"cannot locate {module}"
    return pathlib.Path(spec.origin).read_text()


def _rewrite(text: str) -> str:
    """Apply exactly the mechanical rewrite the fork applied."""
    for old in sorted(MODULE_MAP, key=len, reverse=True):
        text = text.replace(old, MODULE_MAP[old])
    text = re.sub(r"\bNeo4jNodeEncoder\b", "CatalogNodeEncoder", text)
    return re.sub(r"\bNeo4jNode\b", "CatalogNode", text)


def _normalised_ast(text: str) -> str:
    """AST dump with the module docstring dropped.

    Formatting, comments and the provenance docstring are not behaviour, so
    comparing dumps keeps the test about the code and nothing else.
    """
    tree = ast.parse(text)
    body = tree.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    return ast.dump(ast.Module(body=body, type_ignores=[]), indent=1)


@pytest.mark.parametrize("fork_module,library_module", sorted(FORKED.items()))
def test_fork_is_verbatim(fork_module: str, library_module: str) -> None:
    """The fork differs from upstream only by import path and node rename."""
    expected = _normalised_ast(_rewrite(_source(library_module)))
    actual = _normalised_ast(_source(fork_module))
    assert actual == expected, (
        f"{fork_module} has drifted from {library_module}. Either the library "
        f"changed upstream and the fork must carry the change, or the fork was "
        f"edited — which Phase 1 does not permit. A deliberate divergence needs "
        f"a DECISIONS.md record and an entry in DIVERGED."
    )


def test_every_catalog_module_is_accounted_for() -> None:
    """No module can be added to gsf/catalog/ without declaring its provenance.

    A new file that is neither pinned to upstream nor listed as a deliberate
    divergence is exactly how an unreviewed edit would slip into a "verbatim"
    fork.
    """
    known = set(FORKED) | set(DIVERGED)
    found = {
        "gsf.catalog."
        + str(path.relative_to(CATALOG_DIR).with_suffix("")).replace("/", ".")
        for path in CATALOG_DIR.rglob("*.py")
        if path.name != "__init__.py" and "tests" not in path.parts
    }
    assert found == known, f"undeclared: {found - known}; stale: {known - found}"


def test_storage_agnostic_modules_do_not_import_neo4j() -> None:
    """Everything above ``store/`` must survive Phase 4 untouched.

    Phase 4 rewrites ``gsf/catalog/store/`` for Postgres and nothing else. A
    ``neo4j`` import — or Cypher — leaking above that boundary is what would
    turn a contained rewrite into a second fork.
    """
    offenders: list[str] = []
    for path in CATALOG_DIR.rglob("*.py"):
        parts = path.relative_to(CATALOG_DIR).parts
        if "store" in parts or "tests" in parts:
            continue
        text = path.read_text()
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(n == "neo4j" or n.startswith("neo4j.") for n in names):
                offenders.append(f"{path.name}: imports {names}")
    assert not offenders, offenders
