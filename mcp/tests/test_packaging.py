# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The package carries what it needs to run outside a source checkout."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

import gsf_mcp
from gsf_mcp.config import DEFAULT_SPEC_PATH
from gsf_mcp.server import ICON_PATH

_PACKAGE_DIR = Path(gsf_mcp.__file__).resolve().parent
_PROJECT_DIR = _PACKAGE_DIR.parent
_CANONICAL_SPEC = _PROJECT_DIR.parent / "docs" / "openapi" / "gsf-api.json"
_CANONICAL_MARK = _PROJECT_DIR.parent / "frontend" / "public" / "favicon.svg"


def _pyproject() -> dict:
    return tomllib.loads((_PROJECT_DIR / "pyproject.toml").read_text(encoding="utf-8"))


def test_spec_ships_inside_the_package() -> None:
    # Resolving it relative to the repo instead is what limited this to a
    # source checkout: an installed package has no docs/ directory above it.
    assert DEFAULT_SPEC_PATH.is_file()
    assert DEFAULT_SPEC_PATH.parent == _PACKAGE_DIR


def test_packaged_spec_matches_the_canonical_one() -> None:
    # `pnpm openapi` writes both copies; this fails if only one was committed.
    if not _CANONICAL_SPEC.is_file():
        pytest.skip("installed outside the repo; nothing to compare against")

    assert DEFAULT_SPEC_PATH.read_bytes() == _CANONICAL_SPEC.read_bytes(), (
        "The packaged spec drifted from docs/openapi/gsf-api.json. "
        "Run `pnpm openapi` and commit both files."
    )


def test_icon_ships_inside_the_package() -> None:
    assert ICON_PATH.is_file()
    assert ICON_PATH.parent == _PACKAGE_DIR


def test_icon_matches_the_frontend_favicon() -> None:
    # Same mark the product shows, so the two cannot drift into different logos.
    if not _CANONICAL_MARK.is_file():
        pytest.skip("installed outside the repo; nothing to compare against")

    assert ICON_PATH.read_bytes() == _CANONICAL_MARK.read_bytes(), (
        "The packaged icon drifted from frontend/public/favicon.svg."
    )


def test_both_assets_are_forced_into_the_wheel() -> None:
    # Neither is a .py file, so hatchling only carries them if named here. The
    # failure is nasty: the wheel builds and installs, then reads from disk at
    # startup and finds nothing.
    forced = _pyproject()["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]

    assert "gsf_mcp/gsf-api.json" in forced
    assert "gsf_mcp/nvidia-mark.svg" in forced


def test_console_script_is_declared() -> None:
    # The whole point of the separate distribution: `gsf-mcp` on PATH, and
    # `uvx gsf-mcp` without installing anything.
    assert _pyproject()["project"]["scripts"] == {"gsf-mcp": "gsf_mcp.__main__:main"}


def test_stays_independent_of_the_backend_distribution() -> None:
    # Depending on gsf-server would drag in ~450 packages and its git
    # dependencies, which is the thing this package exists to avoid.
    names = _pyproject()["project"]["dependencies"]

    assert not any("gsf-server" in name for name in names)
    assert len(names) <= 5


def test_version_is_reported_not_raised() -> None:
    # Running from an uninstalled source tree is a normal way to develop and
    # must not stop the server from starting.
    assert isinstance(gsf_mcp.get_version(), str)
    assert gsf_mcp.get_version()
