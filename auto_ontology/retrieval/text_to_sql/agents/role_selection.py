# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pick, per question, which role column a join path crosses.

When several columns of one table point at the same column of another
(``posts.OwnerUserId`` and ``posts.LastEditorUserId`` both at ``users.Id``),
the stored join paths use one of them, chosen once at ingest. A question about
a post's owner then gets the editor join. Here each such hop is re-pointed at
the role the question's words name, and left on the default when no role is
named more than the others.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from auto_ontology.dal.attributes import fetch_semantic_fk_join_groups

logger = logging.getLogger(__name__)

#: Words shorter than this are ignored, and longer ones are compared on this
#: many leading letters, so "owns" meets "owner" and "edited" meets "editor".
_STEM = 3


def _words(text: str) -> set[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text or "")
    return {
        word[:_STEM]
        for word in re.findall(r"[a-z]+", spaced.lower())
        if len(word) >= _STEM
    }


def _role_groups(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], list[dict]]:
    """``(child table, parent table, parent column)`` -> its role rows, lower-cased keys."""
    groups: dict[tuple[str, str, str], dict[str, dict]] = {}
    for row in rows:
        key = (
            str(row["source_table_name"]).lower(),
            str(row["target_table_name"]).lower(),
            str(row["target_column_name"]).lower(),
        )
        groups.setdefault(key, {})[str(row["source_column_id"])] = row
    return {k: list(v.values()) for k, v in groups.items() if len(v) > 1}


def choose_role(question: str, roles: list[dict[str, Any]]) -> str | None:
    """The role column *question* names, or None when it names none uniquely.

    A role is scored by the question words found in its column name and
    description, leaving out words every role shares ("user", "id") and words
    of the two table names ("school" for ``schools``): those say which tables
    are meant, not which role.
    """
    vocab = {
        str(r["source_column_id"]): _words(
            f"{r['source_column_name']} {r.get('source_column_description') or ''}"
        )
        for r in roles
    }
    shared = set.intersection(*vocab.values()) if vocab else set()
    if roles:
        shared |= _words(
            f"{roles[0]['source_table_name']} {roles[0]['target_table_name']}"
        )
    asked = _words(question)
    scores = {
        r["source_column_name"]: len(
            (vocab[str(r["source_column_id"])] - shared) & asked
        )
        for r in roles
    }
    best = max(scores.values(), default=0)
    winners = [name for name, score in scores.items() if score == best]
    return winners[0] if best > 0 and len(winners) == 1 else None


def apply_question_roles(question: str, join_paths: list[dict[str, Any]]) -> int:
    """Re-point role hops in *join_paths* at the role *question* names.

    Edits the hops in place and returns how many changed.
    """
    databases = {
        hop.get("source_database")
        for path in join_paths
        for hop in path.get("path") or []
        if hop.get("source_database")
    }
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for database in databases:
        try:
            groups.update(_role_groups(fetch_semantic_fk_join_groups(database)))
        except Exception:
            logger.warning("role groups unavailable for %s", database, exc_info=True)
    if not groups:
        return 0

    changed = 0
    cache: dict[tuple[str, str, str], str | None] = {}
    for path in join_paths:
        for hop in path.get("path") or []:
            for child, parent in (("source", "target"), ("target", "source")):
                key = (
                    str(hop.get(f"{child}_table", "")).lower(),
                    str(hop.get(f"{parent}_table", "")).lower(),
                    str(hop.get(f"{parent}_column", "")).lower(),
                )
                if key not in groups:
                    continue
                if key not in cache:
                    cache[key] = choose_role(question, groups[key])
                role = cache[key]
                if role and role != hop.get(f"{child}_column"):
                    logger.info(
                        "role join: %s.%s -> %s.%s for this question",
                        hop.get(f"{child}_table"),
                        hop.get(f"{child}_column"),
                        hop.get(f"{child}_table"),
                        role,
                    )
                    hop[f"{child}_column"] = role
                    changed += 1
                break
    return changed
