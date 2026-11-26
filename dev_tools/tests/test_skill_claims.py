# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The agent skills name operations, permissions, tools, variables, and files
that exist, so an agent following them reaches the product it was told about."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from dev_tools import check_skill_claims
from dev_tools.check_skill_claims import check_all


def test_every_skill_claim_matches_the_code() -> None:
    assert check_skill_claims._skill_files(), "no skill files were scanned"

    findings = check_all()
    assert not findings, "\n".join(str(f) for f in findings)


def test_claim_rules_find_broken_claims_in_a_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    skills = repo / "skills"
    skills.mkdir(parents=True)
    subprocess.run(["git", "init", "--quiet"], cwd=repo, check=True)

    fixture = skills / "fixture.md"
    fixture.write_text(
        """\
GET /api/fixture-only-missing-operation

The obsolete endpoint is `/api/fixture-only-missing-bare-path`.
The permission is `fixtureonly:read`.
The tool is `search_fixture_only_tool()`.
The variable is `FIXTURE_ONLY_MISSING_ENV`.
The source file is `auto_ontology/fixture_only_missing.py`.
The guide is [missing](./fixture-only-missing-link.md).

| PUT | `/api/sql-attributes/{attr_id}` | `catalog:read`, `fixturetable:read` |
""",
        encoding="utf-8",
    )

    check_skill_claims._tracked_mentions.cache_clear()
    try:
        monkeypatch.setattr(check_skill_claims, "REPO", repo)
        monkeypatch.setattr(check_skill_claims, "SKILLS", skills)
        findings = check_all()
    finally:
        check_skill_claims._tracked_mentions.cache_clear()

    observed = {(finding.kind, finding.claim, finding.problem) for finding in findings}
    assert {
        (
            "operation",
            "GET /api/fixture-only-missing-operation",
            "path not in docs/openapi/auto-ontology-api.json",
        ),
        (
            "operation",
            "/api/fixture-only-missing-bare-path",
            "path not in docs/openapi/auto-ontology-api.json",
        ),
        ("permission", "fixtureonly:read", "no operation requires it"),
        (
            "mcp-tool",
            "search_fixture_only_tool",
            "not an MCP tool or a name in the code",
        ),
        (
            "env-var",
            "FIXTURE_ONLY_MISSING_ENV",
            "not read or documented outside skills/",
        ),
        ("path", "auto_ontology/fixture_only_missing.py", "does not exist"),
        ("link", "./fixture-only-missing-link.md", "target does not exist"),
        (
            "permission",
            "PUT /api/sql-attributes/{attr_id} -> catalog:read, fixturetable:read",
            "documented: catalog:edit",
        ),
    } <= observed
