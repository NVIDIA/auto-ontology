# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend SQL against Prisma-owned tables must name their schema.

This is a source-level scan rather than a database test, because the failure it
guards against does not need a database to happen and does not reliably show up
in one.

The history: these tables lived in `public` and every backend statement reached
them unqualified, which worked only because the role's `search_path` resolved
there. Moving them to `frontend` broke all fifteen statements at once — and
mostly broke them *silently*, since the call sites catch their own exceptions.
`is_semantic_compilation_enabled()` began returning False (so semantic
compilation just stopped running), and acronyms and custom prompts began
returning empty. Nothing crashed. The only evidence was one ERROR line in the
ingestion log.

A schema-qualified name cannot fail that way, so the rule is simply that the
qualification is always present — enforced here rather than left to review.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from gsf.infra.postgres import FRONTEND_SCHEMA

#: Tables Prisma owns. Kept in sync with `frontend/prisma/schema.prisma`.
PRISMA_TABLES = (
    "conversations",
    "messages",
    "prompts",
    "configurations",
    "acronyms",
    "conversation_analytics",
)

#: Models in `schema.prisma` that Prisma does **not** own.
#:
#: A GSF-owned table declared there only so `prisma db push` recognises it and
#: leaves it alone -- without the declaration `db push` reads it as drift and
#: offers to drop it on every `pnpm dev`. These stay in `public` with the rest
#: of GSF's tables, so the frontend-schema rule below does not apply to them.
DECLARED_NOT_OWNED = {"SemanticCompilationHistory"}

_GSF_ROOT = Path(__file__).resolve().parents[2]

#: `FROM foo`, `INTO foo`, `UPDATE foo`, `JOIN foo` — the places a bare table
#: name can appear. A qualified reference has a `.` before the name and so does
#: not match.
_UNQUALIFIED = re.compile(
    r"\b(?:FROM|INTO|UPDATE|JOIN)\s+(" + "|".join(PRISMA_TABLES) + r")\b"
)


def _python_sources() -> list[Path]:
    return [
        path
        for path in _GSF_ROOT.rglob("*.py")
        if "tests" not in path.parts and "__pycache__" not in path.parts
    ]


def test_no_unqualified_prisma_table_references() -> None:
    offenders: list[str] = []
    for path in _python_sources():
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            match = _UNQUALIFIED.search(line)
            if match:
                offenders.append(
                    f"{path.relative_to(_GSF_ROOT)}:{number}: "
                    f"unqualified {match.group(1)!r} — "
                    f"use {{FRONTEND_SCHEMA}}.{match.group(1)}"
                )

    assert not offenders, (
        "Backend SQL references a Prisma-owned table without its schema. These "
        "resolve via search_path and will silently read nothing once it "
        "changes:\n  " + "\n  ".join(offenders)
    )


@pytest.mark.parametrize(
    "module",
    [
        "gsf/server/chat/conversation_dal.py",
        "gsf/server/chat/settings_dal.py",
        "gsf/ingestion_service/config.py",
    ],
)
def test_known_readers_qualify_and_interpolate(module: str) -> None:
    """The three known readers must actually interpolate the constant.

    Guards the specific way this can regress silently: writing
    ``"... FROM {FRONTEND_SCHEMA}.acronyms"`` without the ``f`` prefix is a
    perfectly valid string that Postgres rejects at runtime, inside a
    ``try/except`` that swallows it.
    """
    source = (_GSF_ROOT / Path(module).relative_to("gsf")).read_text()
    assert f"{FRONTEND_SCHEMA}." not in source.replace("FRONTEND_SCHEMA", ""), (
        f"{module} hard-codes the schema name; import FRONTEND_SCHEMA instead"
    )
    for line in source.splitlines():
        if "{FRONTEND_SCHEMA}" in line and "#:" not in line:
            # The literal containing it has to be an f-string. Checked per file
            # rather than per line because triple-quoted blocks span lines.
            assert 'f"' in source or "f'" in source, (
                f"{module}: {{FRONTEND_SCHEMA}} appears in a non-f-string"
            )
            break


def test_prisma_table_list_matches_the_prisma_schema() -> None:
    """If a model is added with an @@map, this list has to learn about it.

    Without this the scan above quietly stops covering new tables.
    """
    schema_file = _GSF_ROOT.parent / "frontend" / "prisma" / "schema.prisma"
    if not schema_file.exists():  # backend-only checkout
        pytest.skip("frontend/prisma/schema.prisma not present")

    mapped = set(re.findall(r'@@map\("([^"]+)"\)', schema_file.read_text()))
    # Auth tables (user/session/account/...) are Better Auth's and the backend
    # does not read them, so only assert the ones we claim to cover exist.
    missing = set(PRISMA_TABLES) - mapped
    assert not missing, (
        f"PRISMA_TABLES names tables absent from schema.prisma: {sorted(missing)}"
    )


def test_all_prisma_models_declare_the_frontend_schema() -> None:
    """Every model Prisma owns must carry @@schema("frontend").

    A model without it lands wherever Prisma's default points, which is how the
    tables would drift back apart. The exceptions in `DECLARED_NOT_OWNED` are
    GSF's own tables, which belong in `public` with the rest of them — named
    one by one so a genuinely new frontend model cannot land there quietly.
    """
    schema_file = _GSF_ROOT.parent / "frontend" / "prisma" / "schema.prisma"
    if not schema_file.exists():
        pytest.skip("frontend/prisma/schema.prisma not present")

    text = schema_file.read_text()
    # Each model, paired with the `@@schema` inside its own block. Matched
    # together rather than as two separate scans: counting them separately says
    # a model is missing its annotation but never which one.
    declared = re.findall(
        r'^model\s+(\w+)\s*\{(?:[^{}]|\{[^{}]*\})*?@@schema\("([^"]+)"\)',
        text,
        re.MULTILINE,
    )
    models = re.findall(r"^model\s+(\w+)\s*\{", text, re.MULTILINE)
    assert len(declared) == len(models), (
        f"{len(models)} model(s) but {len(declared)} @@schema annotation(s): "
        f"{sorted(set(models) - {name for name, _ in declared})} missing one"
    )

    misplaced = {
        name: schema
        for name, schema in declared
        if schema != FRONTEND_SCHEMA and name not in DECLARED_NOT_OWNED
    }
    assert not misplaced, (
        f"models declare schemas other than {FRONTEND_SCHEMA!r}: {misplaced}"
    )
    unowned = {name: schema for name, schema in declared if name in DECLARED_NOT_OWNED}
    assert all(schema == "public" for schema in unowned.values()), (
        f"GSF-owned tables must stay in 'public': {unowned}"
    )
