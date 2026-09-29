# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OSRB licensing checks: SPDX headers and the notice files shipped in images."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# The frontend and MCP images are built from their own directories, so each
# carries a copy of the root notice files to put under /licenses/.
NOTICE_FILES = ("LICENSE", "THIRD_PARTY_NOTICES.md", "CONTAINER_THIRD_PARTY_NOTICES.md")
IMAGE_CONTEXTS = ("frontend", "mcp")

# Files that cannot carry a comment header, are generated, or are third-party
# data. A `<file>.license` sidecar also satisfies the check.
EXEMPT_SUFFIXES = {
    ".json",
    ".lock",
    ".sql",
    ".csv",
    ".parquet",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".svg",
    ".woff",
    ".woff2",
    ".ttf",
    ".whl",
    ".pdf",
    ".license",
}
EXEMPT_NAMES = {
    "LICENSE",
    "pnpm-lock.yaml",
    "uv.lock",
    ".python-version",
    "next-env.d.ts",
}
EXEMPT_PREFIXES = ("vendor/",)


def _tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    return [line for line in out.splitlines() if line]


def _needs_header(path: str) -> bool:
    p = Path(path)
    return not (
        p.suffix.lower() in EXEMPT_SUFFIXES
        or p.name in EXEMPT_NAMES
        or path.startswith(EXEMPT_PREFIXES)
        or (ROOT / f"{path}.license").exists()
    )


def test_every_source_file_has_an_spdx_header() -> None:
    try:
        files = _tracked_files()
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    missing = []
    for path in files:
        full = ROOT / path
        if not _needs_header(path) or not full.is_file():
            continue
        head = full.read_bytes()[:2000].decode("utf-8", errors="ignore")
        if "SPDX-License-Identifier" not in head:
            missing.append(path)
    assert not missing, (
        "Add the NVIDIA SPDX header (see CONTRIBUTING.md) to:\n  "
        + "\n  ".join(missing)
    )


@pytest.mark.parametrize("context", IMAGE_CONTEXTS)
@pytest.mark.parametrize("name", NOTICE_FILES)
def test_image_context_notice_copies_match_root(context: str, name: str) -> None:
    root_copy = (ROOT / name).read_bytes()
    context_copy = ROOT / context / name
    assert context_copy.is_file() and context_copy.read_bytes() == root_copy, (
        f"{context}/{name} must match the root {name}; run: cp {name} {context}/{name}"
    )
