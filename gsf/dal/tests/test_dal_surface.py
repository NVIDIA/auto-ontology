# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Freeze the DAL's public function surface.

``gsf.dal`` is consumed by ``gsf.semantic``, ``gsf.retrieval``, eleven routers
and the frontend. The likeliest way to break one of them is not a broken query
but a function quietly changed with a renamed keyword or a dropped pass-through
argument — which type checks fine and fails at runtime for one caller.

This snapshots every public callable each DAL module defines, with its
signature, and fails when the two disagree.

**Functions only, deliberately.** Module-level constants are not frozen.

Regenerate after an *intentional* signature change::

    uv run python -m gsf.dal.tests.test_dal_surface
"""

from __future__ import annotations

import importlib
import inspect
import json
import pkgutil
import re
from pathlib import Path
from typing import Any

import pytest

import gsf.dal

SNAPSHOT = Path(__file__).with_name("dal_surface.json")


def _dal_modules() -> list[str]:
    """Every domain module directly under ``gsf.dal``.

    Packages are skipped, which covers ``tests``. ``gsf.dal`` itself is not a
    module here, so ``close_store`` is out of scope: it is a lifecycle hook, not
    a data function.
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
        return _stable(str(inspect.signature(obj)))
    except (ValueError, TypeError):  # C-implemented or otherwise opaque
        return "callable(?)"


#: ``<... object at 0x7f...>`` — a default argument that is an object without a
#: ``__repr__`` renders its address, which differs every process. A snapshot
#: containing one can never match itself, so the address is normalised away and
#: the type name kept. ``sql_fragments`` defaults to a ``Table``, whose repr is
#: full of these.
_ADDRESS = re.compile(r" object at 0x[0-9a-f]+")


def _stable(signature: str) -> str:
    return _ADDRESS.sub(" object", signature)


def _public_surface(module: Any) -> dict[str, str]:
    """Public callables *defined in* ``module``, mapped to their shape.

    The ``__module__`` check drops re-exports — helpers imported into a module
    are not part of that module's own contract.
    """
    exported = set(getattr(module, "__all__", ()) or ())
    return {
        name: _describe(obj)
        for name, obj in vars(module).items()
        if not name.startswith("_")
        and callable(obj)
        and (
            getattr(obj, "__module__", None) == module.__name__
            # A module may deliberately re-export something defined elsewhere
            # (``exploration.fetch_table_zones_map`` lives in ``zones``). Those
            # are part of its contract, and __module__ points at the definition,
            # so ``__all__`` is what makes them visible here.
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
        f"uv run python -m gsf.dal.tests.test_dal_surface"
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


if __name__ == "__main__":
    SNAPSHOT.write_text(json.dumps(capture(), indent=2, sort_keys=True) + "\n")
    surface = capture()
    print(
        f"Wrote {SNAPSHOT.relative_to(Path.cwd())}: "
        f"{len(surface)} modules, {sum(len(f) for f in surface.values())} functions"
    )
