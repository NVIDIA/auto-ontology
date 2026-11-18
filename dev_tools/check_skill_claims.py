# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the claims the agent skills make against the code they describe.

The skills in ``skills/`` name REST operations, permissions, MCP tools,
environment variables, and repository files. Each of those has one source of
truth in this repository, so a skill that drifts from the code can be caught
mechanically:

- ``METHOD /api/...`` and bare ``/api/...`` paths: the OpenAPI document;
- ``resource:action`` permissions, and the permission paired with an operation
  in a table row: that operation's ``x-auto-ontology-permissions``;
- MCP tool names: the names the MCP server registers;
- ``UPPER_CASE`` environment variables: anywhere in the tracked source;
- repository paths and relative Markdown links: the working tree.

Run ``python dev_tools/check_skill_claims.py`` for a report; the test in
``dev_tools/tests/test_skill_claims.py`` fails when any claim is wrong.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SKILLS = REPO / "skills"
OPENAPI = REPO / "docs" / "openapi" / "auto-ontology-api.json"
MCP_SOURCES = REPO / "mcp" / "auto_ontology_mcp"

# Generated or signed files describe an evaluation run, not the product.
_SKIPPED_NAMES = {"BENCHMARK.md", "skill-card.md", "skill.oms.sig"}
_METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE")

_OPERATION = re.compile(r"\b(GET|POST|PUT|PATCH|DELETE)\s+`?(/api/[^\s`'\")|,;]*)")
_BARE_PATH = re.compile(r"`(/api/[^\s`]*)`")
_PERMISSION = re.compile(r"`([a-z][A-Za-z]+:[a-z][A-Za-z]+)`")
_TOOL_LIKE = re.compile(
    r"`((?:search|get|check|ask|describe|list|run|find)_[a-z_]+)(?:\(\))?`"
)
_ENV_VAR = re.compile(r"`([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)(?:=[^`]*)?`")
_REPO_PATH = re.compile(
    r"`((?:auto_ontology|mcp|frontend|dev_tools|docs|deploy|helm|skills|scripts)"
    r"/[\w./-]+|[\w-]+\.(?:md|yml|yaml|sh|json|toml))`"
)
_LINK = re.compile(r"\]\(((?!https?://|mailto:|#)[^)\s]+)\)")
_TABLE_ROW = re.compile(
    r"^\|\s*(GET|POST|PUT|PATCH|DELETE)\s*\|\s*`?(/api/[^|`]+)`?\s*\|([^|]*)\|"
)


@dataclass(frozen=True)
class Finding:
    file: str
    line: int
    kind: str
    claim: str
    problem: str

    def __str__(self) -> str:
        return f"{self.file}:{self.line}: [{self.kind}] {self.claim}: {self.problem}"


def _template(path: str) -> str:
    """Normalize ``{term_id}`` / ``{id}`` / ``<id>`` to one placeholder."""
    path = path.split("?", 1)[0].rstrip("/.") or "/"
    return re.sub(r"\{[^}]*\}|<[^>]*>|:[a-z_]+(?=/|$)", "{}", path)


@lru_cache(maxsize=1)
def _openapi() -> dict[str, dict[str, set[str]]]:
    """``{normalized path: {METHOD: {"resource:action", ...}}}``."""
    spec = json.loads(OPENAPI.read_text())
    out: dict[str, dict[str, set[str]]] = {}
    for path, ops in spec["paths"].items():
        methods = out.setdefault(_template(path), {})
        for method, op in ops.items():
            if method.upper() not in _METHODS or not isinstance(op, dict):
                continue
            perms = op.get("x-auto-ontology-permissions") or {}
            if not isinstance(perms, dict):  # "public"
                methods[method.upper()] = set()
                continue
            methods[method.upper()] = {
                f"{resource}:{action}"
                for resource, actions in perms.items()
                for action in actions
            }
    return out


@lru_cache(maxsize=1)
def _all_permissions() -> set[str]:
    return {p for ops in _openapi().values() for perms in ops.values() for p in perms}


@lru_cache(maxsize=1)
def _mcp_tools() -> set[str]:
    names: set[str] = set()
    for source in MCP_SOURCES.glob("*.py"):
        names.update(re.findall(r'\bname="([a-z_]+)"', source.read_text()))
    return names


@lru_cache(maxsize=None)
def _tracked_mentions(token: str) -> bool:
    """Whether *token* appears as a whole word in a tracked file outside skills/."""
    result = subprocess.run(
        ["git", "grep", "-q", "-w", "-F", token, "--", ".", ":!skills"],
        cwd=REPO,
        check=False,
    )
    return result.returncode == 0


def _skill_files() -> list[Path]:
    files = []
    for path in sorted(SKILLS.rglob("*")):
        if not path.is_file() or path.name in _SKIPPED_NAMES:
            continue
        if path.suffix in {".md", ".yaml", ".yml", ".json"}:
            files.append(path)
    return files


def _check_operation(method: str | None, path: str) -> str | None:
    ops = _openapi().get(_template(path))
    if ops is None:
        return "path not in docs/openapi/auto-ontology-api.json"
    if method and method not in ops:
        return f"{method} not documented; documented: {', '.join(sorted(ops))}"
    return None


def check_file(path: Path) -> list[Finding]:
    rel = str(path.relative_to(REPO))
    findings: list[Finding] = []

    def add(line: int, kind: str, claim: str, problem: str) -> None:
        findings.append(Finding(rel, line, kind, claim, problem))

    previous = ""
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        context, previous = previous, line
        claimed_ops = set()
        for method, api_path in _OPERATION.findall(line):
            claimed_ops.add(api_path)
            if "..." in api_path:
                continue
            if problem := _check_operation(method, api_path):
                add(number, "operation", f"{method} {api_path}", problem)
        for api_path in _BARE_PATH.findall(line):
            if api_path in claimed_ops or "..." in api_path:
                continue
            if problem := _check_operation(None, api_path):
                add(number, "operation", api_path, problem)

        if row := _TABLE_ROW.match(line.strip()):
            method, api_path, perm_cell = row.groups()
            api_path = api_path.strip()
            if problem := _check_operation(method, api_path):
                add(number, "operation", f"{method} {api_path}", problem)
            ops = _openapi().get(_template(api_path), {})
            documented = ops.get(method)
            claimed = set(re.findall(r"[a-z][A-Za-z]+:[a-z][A-Za-z]+", perm_cell))
            for perm in sorted(claimed - _all_permissions()):
                add(number, "permission", perm, "no operation requires it")
            if documented is not None and claimed and not documented <= claimed:
                add(
                    number,
                    "permission",
                    f"{method} {api_path} -> {', '.join(sorted(claimed))}",
                    f"documented: {', '.join(sorted(documented)) or '(none)'}",
                )

        for perm in _PERMISSION.findall(line):
            if perm not in _all_permissions():
                add(number, "permission", perm, "no operation requires it")

        for tool in _TOOL_LIKE.findall(line):
            if tool not in _mcp_tools() and not _tracked_mentions(tool):
                add(number, "mcp-tool", tool, "not an MCP tool or a name in the code")

        for var in _ENV_VAR.findall(line):
            if not _tracked_mentions(var):
                add(number, "env-var", var, "not read or documented outside skills/")

        for match in _REPO_PATH.finditer(line):
            # "There is no `x/main.py`" warns an agent off a path on purpose.
            before = f"{context} {line[: match.start()]}"
            if re.search(r"\b(?:no|not)\s*$", before):
                continue
            candidate = match.group(1).rstrip("/.")
            if "*" in candidate or "{" in candidate or "<" in candidate:
                continue
            exists = (REPO / candidate).exists() or (path.parent / candidate).exists()
            if not exists and "/" not in candidate:
                exists = any(REPO.rglob(candidate))
            if not exists:
                add(number, "path", candidate, "does not exist")

        for link in _LINK.findall(line):
            target = link.split("#", 1)[0]
            if target and not (path.parent / target).resolve().exists():
                add(number, "link", link, "target does not exist")

    # A table row's permission is read by the row rule and the prose rule.
    return list(dict.fromkeys(findings))


def check_all() -> list[Finding]:
    return [finding for path in _skill_files() for finding in check_file(path)]


def main() -> int:
    findings = check_all()
    for finding in findings:
        print(finding)
    print(f"{len(findings)} finding(s) across {len(_skill_files())} skill files")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
