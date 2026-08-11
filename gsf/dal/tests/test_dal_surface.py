# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Freeze the DAL's public function surface.

The Neo4j-to-Postgres cutover works by keeping every ``gsf.dal`` function
signature identical and swapping the bodies, so consumers — ``gsf.semantic``,
``gsf.retrieval``, eleven routers, the frontend — never learn which store is
underneath. That decision is only worth anything if it's enforced: the likeliest
way to break the cutover is not a broken query but a function quietly ported
with a renamed keyword or a dropped pass-through argument, which type checks
fine and fails at runtime for one caller.

This snapshots every public callable each DAL module defines, with its
signature, and fails when the two disagree. Once ``gsf/dal/pg/`` and
``gsf/dal/neo4j/`` exist side by side it also asserts the two implementations
expose the same surface.

**Functions only, deliberately.** Module-level constants are not frozen —
several of them (``TABLE_COUNTS_SUBQUERY``, the ``cypher_fragments`` builders'
output) are Cypher, and Cypher is exactly what this refactor is meant to
delete. Freezing them would fight the work rather than protect it.

Regenerate after an *intentional* signature change::

    uv run python -m gsf.dal.tests.test_dal_surface

and say why in ``docs/refactor/drop-neo4j/PROGRESS.md``.
"""

from __future__ import annotations

import importlib
import inspect
import json
import pkgutil
from pathlib import Path
from typing import Any

import pytest

import gsf.dal

SNAPSHOT = Path(__file__).with_name("dal_surface.json")

#: Functions whose *annotations* legitimately differ between backends, with the
#: reason. Parameter **names** are still required to match — those are what
#: callers pass — so this permits a changed type, never a changed call shape.
#:
#: Anything added here needs a DECISIONS.md record first.
BACKEND_SPECIFIC: dict[tuple[str, str], str] = {
    ("users", "resolve_table_filter"): (
        "returns a SQLAlchemy predicate where Neo4j returns a Cypher WHERE "
        "string, and takes a Column rather than a column-name string. Every "
        "caller is inside gsf/dal and is ported with its own phase, so the two "
        "shapes never meet -- DECISION-008"
    ),
}


def _dal_modules() -> list[str]:
    """Every domain module directly under ``gsf.dal``.

    Packages are skipped, which covers ``tests`` now and ``pg`` / ``neo4j``
    once the split lands — those are compared against each other by
    :func:`test_postgres_and_neo4j_surfaces_match` instead. ``gsf.dal`` itself
    is not a module here, so ``close_store`` is out of scope: it's a lifecycle
    hook, not a data function, and Phase 3 deliberately changes what it closes.
    """
    return sorted(m.name for m in pkgutil.iter_modules(gsf.dal.__path__) if not m.ispkg)


def _describe(obj: Any) -> str:
    """How ``obj`` presents itself to a caller.

    Classes are recorded by their bases rather than a signature: the DAL's
    public classes are exceptions and dataclasses, and what a caller depends on
    is ``except UnknownDatabaseIdsError`` still catching the same thing. Some —
    exceptions defined with no body — have no introspectable signature at all.
    """
    if inspect.isclass(obj):
        bases = ", ".join(base.__name__ for base in obj.__bases__)
        return f"class({bases})"
    try:
        return str(inspect.signature(obj))
    except (ValueError, TypeError):  # C-implemented or otherwise opaque
        return "callable(?)"


def _public_surface(module: Any) -> dict[str, str]:
    """Public callables *defined in* ``module``, mapped to their shape.

    The ``__module__`` check drops re-exports — ``Labels``, ``graph``,
    ``get_neo4j_conn`` and friends are imported into most DAL modules and are
    not part of any module's own contract.
    """
    exported = set(getattr(module, "__all__", ()) or ())
    return {
        name: _describe(obj)
        for name, obj in vars(module).items()
        if not name.startswith("_")
        and callable(obj)
        and (
            getattr(obj, "__module__", None) == module.__name__
            # Backend selectors re-export their implementation's functions, so
            # __module__ points at gsf.dal.neo4j.X or gsf.dal.pg.X. Without this
            # a selector would appear to have no public surface at all, and the
            # freeze would silently stop guarding the module it was written for.
            or name in exported
        )
    }


def capture() -> dict[str, dict[str, str]]:
    """The current surface of every DAL module."""
    return {
        name: _public_surface(importlib.import_module(f"gsf.dal.{name}"))
        for name in _dal_modules()
    }


def _load_snapshot() -> dict[str, dict[str, str]]:
    return json.loads(SNAPSHOT.read_text())


@pytest.mark.parametrize("module_name", _dal_modules())
def test_module_surface_is_unchanged(module_name: str) -> None:
    """Each module still exposes exactly the functions it did, unchanged."""
    expected = _load_snapshot()
    assert module_name in expected, (
        f"gsf.dal.{module_name} is new. Regenerate the snapshot and record why "
        f"in PROGRESS.md: uv run python -m gsf.dal.tests.test_dal_surface"
    )

    actual = _public_surface(importlib.import_module(f"gsf.dal.{module_name}"))
    frozen = expected[module_name]

    assert set(actual) - set(frozen) == set(), (
        f"gsf.dal.{module_name} gained public functions. Adding to the DAL is "
        f"fine — regenerate the snapshot."
    )
    assert set(frozen) - set(actual) == set(), (
        f"gsf.dal.{module_name} lost public functions its consumers may call. "
        f"If the removal is intentional, regenerate the snapshot."
    )

    changed = {
        name: f"{frozen[name]} -> {actual[name]}"
        for name in frozen
        if frozen[name] != actual[name]
    }
    assert not changed, (
        f"gsf.dal.{module_name} signatures changed. A renamed keyword or a "
        f"dropped argument breaks callers silently: {changed}"
    )


def test_no_module_disappeared() -> None:
    """A whole domain going missing should fail loudly, not vacuously pass."""
    assert set(_load_snapshot()) == set(_dal_modules())


def test_snapshot_is_not_empty() -> None:
    """Guard against a regeneration that captured nothing."""
    snapshot = _load_snapshot()
    assert sum(len(fns) for fns in snapshot.values()) > 100


def test_postgres_and_neo4j_surfaces_match() -> None:
    """Both implementations must expose an identical surface.

    Skipped until ``gsf/dal/pg/`` and ``gsf/dal/neo4j/`` exist side by side
    (Phase 3 onward). From then on this is the check that makes the
    ``GSF_STORE`` flip safe: the two packages have to be substitutable
    function-for-function, or flipping the switch breaks a caller.
    """
    pg_root = Path(gsf.dal.__path__[0]) / "pg"
    neo4j_root = Path(gsf.dal.__path__[0]) / "neo4j"
    if not (pg_root.is_dir() and neo4j_root.is_dir()):
        pytest.skip("split implementations do not exist yet (lands in Phase 3)")

    # Only modules with both implementations. A domain reaches pg/ one phase at
    # a time, and session.py / schema.py have no Neo4j counterpart by design.
    both = sorted(
        path.stem
        for path in pg_root.glob("*.py")
        if not path.stem.startswith("_") and (neo4j_root / path.name).is_file()
    )
    assert both, "no domain has been ported yet; this test is vacuous"

    mismatches: dict[str, str] = {}
    for stem in both:
        path = pg_root / f"{stem}.py"
        pg_module = importlib.import_module(f"gsf.dal.pg.{path.stem}")
        neo_module = importlib.import_module(f"gsf.dal.neo4j.{path.stem}")
        pg = _public_surface(pg_module)
        neo = _public_surface(neo_module)

        # An allowed divergence still has to keep its parameter names: a caller
        # passes those, and a renamed keyword breaks it whatever the types say.
        for name in set(pg) & set(neo):
            if (path.stem, name) not in BACKEND_SPECIFIC:
                continue
            pg_params = list(inspect.signature(getattr(pg_module, name)).parameters)
            neo_params = list(inspect.signature(getattr(neo_module, name)).parameters)
            assert pg_params == neo_params, (
                f"{path.stem}.{name} is allowed to differ in types, not in "
                f"parameters: {pg_params} vs {neo_params}"
            )
            pg[name] = neo[name] = "<backend-specific>"

        if pg != neo:
            only_pg = {n: pg[n] for n in set(pg) - set(neo)}
            only_neo = {n: neo[n] for n in set(neo) - set(pg)}
            differing = {
                n: f"pg{pg[n]} != neo4j{neo[n]}"
                for n in set(pg) & set(neo)
                if pg[n] != neo[n]
            }
            mismatches[path.stem] = (
                f"only in pg={only_pg}, only in neo4j={only_neo}, differing={differing}"
            )

    assert not mismatches, f"pg/neo4j DAL surfaces diverged: {mismatches}"


if __name__ == "__main__":
    SNAPSHOT.write_text(json.dumps(capture(), indent=2, sort_keys=True) + "\n")
    surface = capture()
    print(
        f"Wrote {SNAPSHOT.relative_to(Path.cwd())}: "
        f"{len(surface)} modules, {sum(len(f) for f in surface.values())} functions"
    )
