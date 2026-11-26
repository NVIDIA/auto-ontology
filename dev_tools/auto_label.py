# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Add the labels an issue or pull request can be given mechanically.

Run by ``.github/workflows/auto-label.yml`` when an issue or PR is opened (not
on later edits or pushes, so a label a maintainer removes stays removed). It only ADDS labels, never removes one, and never creates a label: a
label the repository does not have is skipped. Judgment labels (the area
themes ``correctness``, ``lifecycle/workflow`` and so on, and ``enhancement``)
are left to people or to the ``label-auto-ontology`` skill.

Rules:

- ``external``: a PR whose author is outside the core team, by GitHub's
  ``author_association`` (never a bot).
- ``documentation``: a PR that changes only Markdown files and ``docs/`` (the
  generated ``docs/openapi/`` specs and the ``THIRD_PARTY_NOTICES.md`` files
  do not count).
- ``UI``: a PR that changes only ``frontend/`` files, at least one of them a
  stylesheet or component, and no manifest or lockfile (a dependency bump is
  not a UI change).
- ``bug``: a title that starts with ``fix`` or ``bug`` (``fix:``,
  ``fix(scope):``, ``Fix ...``, ``[Bug]: ...``), unless the item already has a
  type label (``bug``, ``enhancement``, ``documentation`` or ``question``) or
  is a documentation-only PR.

Issue and PR text is untrusted: it is only ever matched here, never executed
or interpolated into a command.

Environment: ``REPO`` (``owner/name``), ``NUMBER``, ``GH_TOKEN`` for ``gh``,
and optionally ``DRY_RUN=1`` to print the labels without applying them.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

TYPE_LABELS = frozenset({"bug", "enhancement", "documentation", "question"})
EXTERNAL_ASSOCIATIONS = frozenset(
    {"CONTRIBUTOR", "FIRST_TIME_CONTRIBUTOR", "FIRST_TIMER", "NONE"}
)

_BUG_TITLE = re.compile(r"^\s*(\[bug\]|bug\b|fix\b|fix\()", re.IGNORECASE)
_UI_SOURCE = (".css", ".tsx")
_FRONTEND_MANIFESTS = (
    "package.json",
    "pnpm-lock.yaml",
    "package-lock.json",
    "pnpm-workspace.yaml",
)


@dataclass(frozen=True)
class Item:
    """The parts of an issue or pull request the rules look at."""

    is_pr: bool
    title: str
    author: str
    association: str
    labels: frozenset[str]
    files: tuple[str, ...] = ()


def _is_docs_file(path: str) -> bool:
    if path.startswith("docs/openapi/") or path.endswith("THIRD_PARTY_NOTICES.md"):
        return False
    return path.endswith((".md", ".mdx")) or path.startswith("docs/")


def _is_ui_change(files: Sequence[str]) -> bool:
    if not files or not all(f.startswith("frontend/") for f in files):
        return False
    if any(f.rsplit("/", 1)[-1] in _FRONTEND_MANIFESTS for f in files):
        return False
    return any(f.endswith(_UI_SOURCE) for f in files)


def labels_to_add(item: Item, available: Iterable[str]) -> list[str]:
    """The labels *item* should gain, in a stable order.

    *available* is the repository's label set; anything outside it is dropped.
    """
    wanted: list[str] = []
    has_type = bool(item.labels & TYPE_LABELS)
    docs_only = (
        item.is_pr and bool(item.files) and all(_is_docs_file(f) for f in item.files)
    )

    if (
        item.is_pr
        and item.association in EXTERNAL_ASSOCIATIONS
        and not item.author.endswith("[bot]")
    ):
        wanted.append("external")
    if docs_only and not has_type:
        wanted.append("documentation")
    if item.is_pr and _is_ui_change(item.files):
        wanted.append("UI")
    if not has_type and not docs_only and _BUG_TITLE.match(item.title):
        wanted.append("bug")

    known = set(available)
    return [label for label in wanted if label in known and label not in item.labels]


def _gh(*args: str) -> str:
    result = subprocess.run(["gh", *args], check=True, capture_output=True, text=True)
    return result.stdout


def fetch_item(repo: str, number: int) -> Item:
    """Read one issue or PR (the issues endpoint serves both)."""
    data = json.loads(_gh("api", f"repos/{repo}/issues/{number}"))
    is_pr = "pull_request" in data
    files: tuple[str, ...] = ()
    if is_pr:
        listing = _gh(
            "api",
            "--paginate",
            f"repos/{repo}/pulls/{number}/files",
            "--jq",
            ".[].filename",
        )
        files = tuple(line for line in listing.splitlines() if line)
    return Item(
        is_pr=is_pr,
        title=data.get("title") or "",
        author=(data.get("user") or {}).get("login") or "",
        association=data.get("author_association") or "",
        labels=frozenset(label["name"] for label in data.get("labels", [])),
        files=files,
    )


def repo_labels(repo: str) -> set[str]:
    out = _gh("api", "--paginate", f"repos/{repo}/labels", "--jq", ".[].name")
    return {line for line in out.splitlines() if line}


def apply_labels(repo: str, number: int, labels: Sequence[str]) -> None:
    """Add *labels* (POST adds; it does not replace the ones already there)."""
    subprocess.run(
        [
            "gh",
            "api",
            "-X",
            "POST",
            f"repos/{repo}/issues/{number}/labels",
            "--input",
            "-",
        ],
        input=json.dumps({"labels": list(labels)}),
        check=True,
        capture_output=True,
        text=True,
    )


def main() -> int:
    repo = os.environ["REPO"]
    number = int(os.environ["NUMBER"])
    dry_run = os.environ.get("DRY_RUN", "") not in ("", "0", "false")

    try:
        item = fetch_item(repo, number)
        to_add = labels_to_add(item, repo_labels(repo))
        kind = "PR" if item.is_pr else "issue"
        if not to_add:
            print(f"{kind} #{number}: nothing to add")
            return 0
        print(f"{kind} #{number}: add {', '.join(to_add)}")
        if not dry_run:
            apply_labels(repo, number, to_add)
    except subprocess.CalledProcessError as exc:
        # Labels are cosmetic: say what failed, but do not turn the PR red.
        detail = (exc.stderr or "").strip().replace("\n", " ")
        print(f"::warning::auto-label could not finish for #{number}: {detail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
