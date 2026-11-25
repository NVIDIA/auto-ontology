# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The mechanical labeling rules in ``dev_tools/auto_label.py``."""

from __future__ import annotations

import pytest

from dev_tools.auto_label import Item, labels_to_add

AVAILABLE = {
    "bug",
    "enhancement",
    "documentation",
    "question",
    "external",
    "UI",
    "correctness",
}


def pr(
    title: str = "Some change",
    *,
    files: tuple[str, ...] = ("auto_ontology/x.py",),
    association: str = "MEMBER",
    author: str = "someone",
    labels: frozenset[str] = frozenset(),
) -> Item:
    return Item(True, title, author, association, labels, files)


def issue(title: str, labels: frozenset[str] = frozenset()) -> Item:
    return Item(False, title, "someone", "NONE", labels)


@pytest.mark.parametrize(
    "association", ["CONTRIBUTOR", "FIRST_TIME_CONTRIBUTOR", "FIRST_TIMER", "NONE"]
)
def test_outside_authors_get_external(association: str) -> None:
    assert labels_to_add(pr(association=association), AVAILABLE) == ["external"]


@pytest.mark.parametrize("association", ["MEMBER", "OWNER", "COLLABORATOR"])
def test_core_team_authors_do_not(association: str) -> None:
    assert labels_to_add(pr(association=association), AVAILABLE) == []


def test_bots_and_issues_never_get_external() -> None:
    bot = pr(association="NONE", author="dependabot[bot]")
    assert labels_to_add(bot, AVAILABLE) == []
    assert labels_to_add(issue("Something"), AVAILABLE) == []


@pytest.mark.parametrize(
    "title",
    [
        "fix: stop dropping foreign keys",
        "fix(sqlite): resolve relative paths",
        "Fix review findings on table-derived SQL attributes",
        "[Bug]: plain counting questions misrouted",
        "bug in the registry",
    ],
)
def test_fix_titles_get_bug(title: str) -> None:
    assert labels_to_add(pr(title), AVAILABLE) == ["bug"]
    assert labels_to_add(issue(title), AVAILABLE) == ["bug"]


@pytest.mark.parametrize(
    "title", ["Add an Oracle Database connector", "Prefix handling", "Fixture cleanup"]
)
def test_other_titles_get_no_type(title: str) -> None:
    assert labels_to_add(pr(title), AVAILABLE) == []


def test_an_existing_type_label_is_respected() -> None:
    assert (
        labels_to_add(pr("Fix a thing", labels=frozenset({"enhancement"})), AVAILABLE)
        == []
    )


def test_docs_only_pr_is_documentation() -> None:
    docs = pr(files=("README.md", "docs/connections.md"))
    assert labels_to_add(docs, AVAILABLE) == ["documentation"]


def test_generated_openapi_or_code_is_not_documentation() -> None:
    assert labels_to_add(pr(files=("docs/openapi/backend.json",)), AVAILABLE) == []
    assert labels_to_add(pr(files=("README.md", "auto_ontology/x.py")), AVAILABLE) == []


def test_third_party_notices_are_not_documentation() -> None:
    files = ("THIRD_PARTY_NOTICES.md", "mcp/THIRD_PARTY_NOTICES.md")
    assert labels_to_add(pr(files=files), AVAILABLE) == []


def test_docs_only_fix_is_documentation_not_bug() -> None:
    item = pr("Fix typos in the README", files=("README.md",))
    assert labels_to_add(item, AVAILABLE) == ["documentation"]


def test_frontend_style_pr_is_ui() -> None:
    files = ("frontend/app/globals.css", "frontend/common/SidePanelLayout.tsx")
    assert labels_to_add(pr(files=files), AVAILABLE) == ["UI"]


def test_frontend_dependency_bump_is_not_ui() -> None:
    files = ("frontend/package.json", "frontend/pnpm-lock.yaml")
    assert labels_to_add(pr("Bump Next.js", files=files), AVAILABLE) == []


def test_mixed_frontend_and_backend_is_not_ui() -> None:
    files = ("frontend/common/A.tsx", "auto_ontology/server/x.py")
    assert labels_to_add(pr(files=files), AVAILABLE) == []


def test_labels_are_combined_and_unknown_ones_dropped() -> None:
    item = pr(
        "fix: chart colors",
        files=("frontend/app/globals.css",),
        association="FIRST_TIME_CONTRIBUTOR",
    )
    assert labels_to_add(item, AVAILABLE) == ["external", "UI", "bug"]
    assert labels_to_add(item, {"bug"}) == ["bug"]


def test_labels_already_present_are_not_repeated() -> None:
    item = pr(association="NONE", labels=frozenset({"external"}))
    assert labels_to_add(item, AVAILABLE) == []
