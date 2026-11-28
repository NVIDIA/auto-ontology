# SPDX-FileCopyrightText: Copyright (c) 2026 Sachingouda Patil
# SPDX-License-Identifier: Apache-2.0

"""Guardrails ensuring shell scripts and Dockerfiles keep LF line endings."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _tracked_lf_required_files() -> list[str]:
    try:
        out = subprocess.run(
            ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    matched: list[str] = []
    for rel in out.splitlines():
        if not rel:
            continue
        name = Path(rel).name
        if (
            name.endswith(".sh")
            or name == "Dockerfile"
            or name.startswith("Dockerfile.")
            or name.endswith(".dockerfile")
        ):
            matched.append(rel)
    return matched


def test_shell_scripts_and_dockerfiles_enforce_lf_in_gitattributes() -> None:
    files = _tracked_lf_required_files()
    assert files, "expected at least one tracked Dockerfile or shell script"
    out = subprocess.run(
        ["git", "check-attr", "eol", "--", *files],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    non_lf = [
        line
        for line in out.splitlines()
        if line and not line.rstrip().endswith(": eol: lf")
    ]
    assert not non_lf, "Add `eol=lf` rules in .gitattributes for:\n  " + "\n  ".join(
        non_lf
    )


def test_shell_scripts_and_dockerfiles_have_no_crlf() -> None:
    files = _tracked_lf_required_files()
    assert files, "expected at least one tracked Dockerfile or shell script"
    with_crlf = [
        rel
        for rel in files
        if (ROOT / rel).is_file() and b"\r\n" in (ROOT / rel).read_bytes()
    ]
    assert not with_crlf, (
        "Shell scripts and Dockerfiles must use LF line endings:\n  "
        + "\n  ".join(with_crlf)
    )
