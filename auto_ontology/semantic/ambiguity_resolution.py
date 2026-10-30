# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tell apart ColumnAttributes that share a name across tables of one database.

Runs last in compilation, after description enrichment. Attributes are grouped
by normalized name (``points``, ``Points`` and ``POINTS`` are one name); a group
is two or more attributes in different tables. One LLM call per group writes
each member's ``ambiguity_resolver``: when to use it rather than the others,
without restating its description. Every other attribute of the database gets
``''``.

Seeing the members side by side also exposes descriptions that mislead — a
running total described as a per-event value — and catalog descriptions that
say nothing, being empty or only the column's name. The same call returns
corrected text for the first and a written one for the second, applied to
whichever description needs it:

* the attribute's semantic description, through
  :func:`auto_ontology.dal.descriptions.write_attribute_descriptions`, so a
  certified attribute gets a suggestion instead;
* the catalog column's description, unless it is certified, through the same
  path as an edit in the UI, which re-embeds what that text feeds.

The whole database runs every time, since a newly compiled table can make an
existing name ambiguous.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from auto_ontology.dal.ambiguity import (
    fetch_attributes_for_ambiguity,
    write_ambiguity_resolvers,
)
from auto_ontology.dal.descriptions import write_attribute_descriptions
from auto_ontology.semantic.cancellation import is_cancelled
from auto_ontology.server.datasources.service import update_node_properties
from auto_ontology.semantic.models import AmbiguityResolver, AmbiguityResolversResult
from auto_ontology.utils.llm_invoke import (
    get_llm_client,
    invoke_with_structured_output,
)
from auto_ontology.utils.sample_values import stringify_sample_values

logger = logging.getLogger(__name__)

_WORKERS = 3
_MAX_SAMPLES = 5
_MAX_SAMPLE_LEN = 40

_SYSTEM_PROMPT = """\
You document a relational database for a semantic layer that a text-to-SQL \
agent reads. The columns listed below share a name but live in different \
tables. For each one, write when to use it instead of the others, and correct \
its descriptions when they mislead.

Rules for the resolver:
- Say what sets this column apart from the others: the grain of its table \
(one row per what), the event or moment it records, whether it is cumulative \
or per occurrence, or which question wording points to it.
- Do not restate the column's description; the agent already reads it. Write \
only what tells the columns apart.
- One or two plain sentences per column. No SQL, no speculation beyond the \
evidence.
- Each note must read as a standalone fact. Never refer to this prompt or say \
that something was or was not provided.

Rules for the fixes:
- A description is misleading when it contradicts the grain of its table, the \
samples, or what the other columns show, or when it describes this column in \
words that fit another member better.
- When the catalog description is misleading, return the corrected text in \
catalog_description_fix; when the semantic description is, in \
description_fix. Keep everything accurate from the original, including every \
value code, and change only what is wrong.
- A column whose catalog description reads "(missing)" needs one: always \
write it in catalog_description_fix, one plain sentence on what the column \
holds, in words that set it apart from the others.
- Otherwise leave a fix empty when its description is accurate or merely \
terse. Never reword accurate text.

Return exactly one entry per listed column, using its table.column as written."""


def normalized_name(name: str) -> str:
    """Lower-case words of *name* in order: camelCase, snake and spaces."""
    spaced = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", name or "")
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", spaced)
    return " ".join(t.lower() for t in re.split(r"[^A-Za-z0-9]+", spaced) if t)


def group_ambiguous(attributes: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Groups of same-named attributes spanning at least two tables."""
    by_name: dict[str, list[dict[str, Any]]] = {}
    for attribute in attributes:
        key = normalized_name(attribute.get("name") or "")
        if key:
            by_name.setdefault(key, []).append(attribute)
    return [
        members
        for members in by_name.values()
        if len({str(m["table_id"]) for m in members}) > 1
    ]


def resolve_ambiguities(database_name: str) -> int:
    """Write the resolvers of *database_name*; returns how many were written.

    Also applies the description fixes the same calls return.
    """
    attributes = fetch_attributes_for_ambiguity(database_name)
    groups = group_ambiguous(attributes)
    logger.info(
        "Ambiguity resolution (database=%r): %d ambiguous name(s) over %d attribute(s)",
        database_name,
        len(groups),
        sum(len(g) for g in groups),
    )

    entries: dict[str, AmbiguityResolver] = {}
    with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
        futures = [pool.submit(_resolve_one, group) for group in groups]
        for future in as_completed(futures):
            entries.update(future.result())
    if is_cancelled():
        logger.info("Ambiguity resolution cancelled — nothing written")
        return 0

    resolvers = {
        aid: e.resolver.strip() for aid, e in entries.items() if e.resolver.strip()
    }
    written = write_ambiguity_resolvers(database_name, resolvers)
    by_id = {str(a["id"]): a for a in attributes}
    attribute_fixes = _changed(entries, by_id, "description_fix", "description")
    catalog_fixes = _changed(
        entries, by_id, "catalog_description_fix", "column_description"
    )
    fixed_attributes = write_attribute_descriptions(attribute_fixes)
    fixed_columns = _write_catalog_fixes(catalog_fixes, by_id)
    logger.info(
        "Ambiguity resolution finished (database=%r): %d resolver(s) written, "
        "%d attribute and %d catalog description(s) corrected",
        database_name,
        written,
        len(fixed_attributes),
        fixed_columns,
    )
    return written


def _resolve_one(group: list[dict[str, Any]]) -> dict[str, AmbiguityResolver]:
    if is_cancelled():
        return {}
    try:
        result = invoke_with_structured_output(
            get_llm_client(temperature=0.0),
            [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=build_group_prompt(group)),
            ],
            AmbiguityResolversResult,
        )
    except Exception:
        logger.warning(
            "Ambiguity resolution errored for %r — skipping it",
            group[0].get("name"),
            exc_info=True,
        )
        return {}
    if result is None:
        logger.warning(
            "Ambiguity resolution returned no result for %r — skipping it",
            group[0].get("name"),
        )
        return {}
    return map_members(result, group)


def build_group_prompt(group: list[dict[str, Any]]) -> str:
    """One block per member: its table, descriptions and samples."""
    blocks = [f"Columns named {group[0].get('name')!r}:"]
    for member in group:
        lines = [f"\n{member_label(member)}"]
        if member.get("table_description"):
            lines.append(f"  table: {member['table_description']}")
        catalog = (member.get("column_description") or "").strip()
        semantic = (member.get("description") or "").strip()
        if is_bare_description(member):
            lines.append("  catalog description: (missing)")
        else:
            lines.append(f"  catalog description: {catalog}")
        if semantic and semantic != catalog:
            lines.append(f"  semantic description: {semantic}")
        if member.get("value_description"):
            lines.append(f"  value description: {member['value_description']}")
        samples = stringify_sample_values(
            member.get("sample_values"), max_len=_MAX_SAMPLE_LEN
        )
        if samples:
            lines.append(f"  samples: {', '.join(samples[:_MAX_SAMPLES])}")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def is_bare_description(member: dict[str, Any]) -> bool:
    """Whether the catalog description is empty or only repeats the name."""
    text = normalized_name(member.get("column_description") or "")
    return text in {
        "",
        normalized_name(member.get("column_name") or ""),
        normalized_name(member.get("name") or ""),
    }


def member_label(member: dict[str, Any]) -> str:
    return f"{member['table_name']}.{member['column_name']}"


def map_members(
    result: AmbiguityResolversResult, group: list[dict[str, Any]]
) -> dict[str, AmbiguityResolver]:
    """Map the model's ``table.column`` entries back to attribute ids.

    Exact label first, then a case-insensitive match with quotes stripped.
    """
    by_label = {member_label(m): str(m["id"]) for m in group}
    loose = {_loose(label): attribute_id for label, attribute_id in by_label.items()}
    out: dict[str, AmbiguityResolver] = {}
    for entry in result.members:
        attribute_id = by_label.get(entry.table_column) or loose.get(
            _loose(entry.table_column)
        )
        if attribute_id:
            out.setdefault(attribute_id, entry)
    return out


def _changed(
    entries: dict[str, AmbiguityResolver],
    by_id: dict[str, dict[str, Any]],
    fix_field: str,
    current_field: str,
) -> dict[str, str]:
    """Fixes keyed by attribute id that differ from the current text."""
    out: dict[str, str] = {}
    for attribute_id, entry in entries.items():
        text = (getattr(entry, fix_field) or "").strip()
        current = (by_id[attribute_id].get(current_field) or "").strip()
        if text and text != current:
            out[attribute_id] = text
    return out


def _write_catalog_fixes(
    fixes: dict[str, str], by_id: dict[str, dict[str, Any]]
) -> int:
    """Write uncertified catalog fixes the way a UI edit does; returns the count."""
    fixed = 0
    for attribute_id, text in fixes.items():
        attribute = by_id[attribute_id]
        if attribute.get("column_certified"):
            continue
        try:
            if update_node_properties(
                str(attribute["column_id"]), {"description": text}
            ):
                fixed += 1
                logger.info(
                    "Corrected catalog description of %s", member_label(attribute)
                )
        except Exception:
            logger.warning(
                "Catalog description fix failed for %s",
                member_label(attribute),
                exc_info=True,
            )
    return fixed


def _loose(label: str) -> str:
    return re.sub(r"[\"'`\s]", "", label or "").lower()
