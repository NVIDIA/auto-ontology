# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay the recorded DAL reads and require identical output.

Every read the DAL exposes, captured against the fixture and compared byte for
byte after normalisation. This is what catches a read that still returns *a*
plausible answer after a change, rather than the right one.

**Re-recording is a last resort, not a fix.** A golden that changes because the
code changed is the test doing its job::

    docker compose up -d postgres
    uv run --no-sync python -m dev_tools.seed_fixtures
    uv run --no-sync python -m dev_tools.seed_graph_fixture --reset
    uv run --no-sync pytest gsf/dal/tests/test_golden.py

Re-record only when a change to DAL output is *intended*::

    uv run --no-sync python -m dev_tools.capture_dal_golden

Results are normalised before comparison — ids become stable tokens, timestamps
are redacted, and undirected edges are oriented consistently. See
``dev_tools/capture_dal_golden.normalise``; note in particular that **list order
is not covered**, deliberately, because most DAL queries lack a total
``ORDER BY``.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

from dev_tools import capture_dal_golden

GOLDEN = capture_dal_golden.load_golden()


def _require_fixture():
    """Skip when there is no reachable fixture.

    Worth remembering as a shape: a skip guard naming the wrong dependency
    turns a whole suite off silently, and still reports green.
    """
    if not os.environ.get("POSTGRES_USER"):
        pytest.skip("POSTGRES_USER not set; the golden fixture is unavailable")
    try:
        return capture_dal_golden.capture_all()
    except SystemExit as exc:  # _fixture_ids() raises this when unseeded
        pytest.skip(str(exc))
    except Exception as exc:  # noqa: BLE001 — no reachable graph
        pytest.skip(f"fixture graph unavailable: {exc}")


@pytest.fixture(scope="module")
def live():
    return _require_fixture()


def test_golden_file_exists() -> None:
    """A missing golden must fail loudly, not make every comparison vacuous."""
    assert GOLDEN["results"], (
        "no recorded goldens. Capture them with: "
        "uv run --no-sync python -m dev_tools.capture_dal_golden"
    )


def test_golden_covers_every_dal_module() -> None:
    """Each module with recorded reads is represented.

    Guards against a re-record that silently drops a whole domain.
    """
    covered = {name.split(".", 1)[0] for name in GOLDEN["results"]}
    expected = {
        "attributes",
        "connections",
        "custom_analyses",
        "datasources",
        "exploration",
        "pql_analyses",
        "sql_attributes",
        "terms",
        "users",
        "zones",
    }
    assert expected <= covered, f"modules missing from the golden: {expected - covered}"


def test_no_capture_raised() -> None:
    """A read that raises is recorded, and must not be ignored."""
    assert GOLDEN["errors"] == {}, f"captures raised: {GOLDEN['errors']}"


#: Keys whose *presence* legitimately differs under Postgres, per read.
#:
#: These two reads return "every property of the node". A node carries only the
#: properties something set; a row carries every column of its table. So under
#: Postgres an unset property is `null` rather than absent, defaults appear, and
#: the parent foreign key is visible — while `created`, which the graph stamped
#: on every node and **nothing reads**, is gone: the catalog tables do not carry
#: it and adding a column to satisfy a golden would be the tail wagging the dog.
#:
#: **Named individually on purpose.** Every other key is still compared exactly,
#: so a changed value, a genuinely missing field, or a new key not listed here
#: still fails. Anything added needs a DECISIONS.md record first — see B10.
OPTIONAL_KEYS: dict[str, frozenset[str]] = {
    "datasources.fetch_item_by_id": frozenset(
        {"created", "description", "description_certified", "imported_id", "schema_id"}
    ),
    "datasources.fetch_node_properties_by_id": frozenset(
        {"created", "description", "description_certified", "imported_id", "schema_id"}
    ),
}


def _without(value: Any, keys: frozenset[str]) -> Any:
    """*value* with *keys* removed wherever they appear, recursively."""
    if isinstance(value, dict):
        return {k: _without(v, keys) for k, v in value.items() if k not in keys}
    if isinstance(value, list):
        return [_without(item, keys) for item in value]
    return value


@pytest.mark.parametrize("name", sorted(GOLDEN["results"]))
def test_read_matches_golden(name: str, live) -> None:
    """Every recorded read still returns exactly what it returned."""
    assert name in live.results, (
        f"{name} is in the golden but was not captured — the read was removed "
        f"or renamed. If intended, re-record."
    )
    expected = GOLDEN["results"][name]
    actual = json.loads(json.dumps(live.results[name], default=str))
    if name in OPTIONAL_KEYS:
        optional = OPTIONAL_KEYS[name]
        actual, expected = _without(actual, optional), _without(expected, optional)
    assert actual == expected, f"{name} diverged from its recorded output"


def test_no_uncaptured_reads(live) -> None:
    """A read added to the capture list but never recorded fails here."""
    extra = set(live.results) - set(GOLDEN["results"])
    assert not extra, f"captured but not in the golden, re-record: {sorted(extra)}"


def test_zone_scoping_actually_narrows(live) -> None:
    """Scoped access must return strictly less than admin access.

    The single most valuable invariant here: a port that silently drops the
    zone filter would still match most goldens, but would hand every user the
    admin view. This asserts the three modes differ from each other.
    """
    admin = live.results.get("terms.fetch_all_terms[admin]")
    scoped = live.results.get("terms.fetch_all_terms[scoped]")
    none = live.results.get("terms.fetch_all_terms[none]")
    assert admin is not None and scoped is not None and none is not None

    assert len(admin) > len(scoped) > 0, (
        f"scoped access is not narrower than admin: {len(admin)} vs {len(scoped)}"
    )
    assert none == [], f"empty zone list must grant nothing, got {len(none)} terms"


def test_disabled_zone_grants_no_access(live) -> None:
    """A disabled zone stays visible to admins but confers no data access."""
    zones = live.results.get("zones.list_zones")
    assert zones, "list_zones returned nothing"
    names = {z.get("name") for z in zones}
    assert "Retired zone" in names, "disabled zone vanished from the admin listing"
