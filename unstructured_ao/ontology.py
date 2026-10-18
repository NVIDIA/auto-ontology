# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the ontology knows about a set of attributes, and how it reads.

Two kinds of thing, both keyed on attributes a search has already found:

- **families** — attributes measuring the same quantity, and what tells them
  apart. A corpus that states one indicator for Austria, for Portugal and for
  2023 holds three genuine figures, and nothing in the passage text says they
  are rivals.
- **relations** — adjudicated pairs: a total and its component, a level and a
  growth rate, the same measure from two publications.

:func:`ontology_note` renders both as the prompt block that goes in front of a
context. It lives here rather than in whatever is doing the retrieving because
every route into this ontology wants the same block: the one-shot attribute
pipeline, the chunk-first pipeline and the agent all used to render it
separately, and the agent's version printed relation names raw with no family
notes at all — so the thing the ontology knows best, that an indicator in the
context has rival readings and here they are, never reached the model.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import Connection, select

from auto_ontology.dal.schema import (
    unstructured_attribute,
    unstructured_attribute__attribute,
    unstructured_attribute_family,
    unstructured_attribute_family__attribute,
)

#: Families whose members genuinely compete. A ``place`` family is a breakdown
#: — Austria's value and Portugal's are both right and answer different
#: questions — so warning that it "reports this more than once" would
#: manufacture a conflict and invite a hedge where none is needed. Only a
#: different period or a different publication puts two readings of the same
#: quantity in front of the model at once.
RIVAL_DIMENSIONS = {"period", "source"}

#: How each relation reads as an instruction, and the order the groups appear
#: in. Rival readings come first because they are the ones that silently
#: produce a confident wrong number; the rest guard against conflating two
#: sources that are merely adjacent.
RELATION_HEADINGS = (
    (
        "same_measure_different_source",
        "The same measure from different publications — the values may "
        "disagree, so say which you used:",
    ),
    (
        "same_measure_different_period",
        "The same measure for different periods — do not read one period's "
        "figure as another's:",
    ),
    (
        "aggregate_and_part",
        "Totals and their components — do not answer about the whole with a "
        "part's figure, or the reverse:",
    ),
    ("inverse", "Measures that move oppositely by construction:"),
    (
        "related_but_distinct",
        "Adjacent but different measures — do not treat them as the same quantity:",
    ),
    ("same_measure_different_place", "The same measure for different places:"),
)


def family_rows(conn: Connection, attribute_ids: list[str]) -> list[dict[str, Any]]:
    """Every member of every family *attribute_ids* touch.

    Whole families, not just the matched members: the point of a family is the
    members that did *not* match, since those are the readings the answer could
    silently be taken from instead.

    ``aggregate_and_part`` groupings are excluded when families are built, not
    here — an EU total and a country value are both right and do not compete.
    """
    if not attribute_ids:
        return []
    member = unstructured_attribute_family__attribute
    statement = (
        select(
            unstructured_attribute_family.c.id.label("family_id"),
            unstructured_attribute_family.c.canonical,
            unstructured_attribute_family.c.dimension,
            unstructured_attribute_family.c.note,
            member.c.attribute_id,
            member.c.qualifier,
            member.c.value,
        )
        .select_from(
            unstructured_attribute_family.join(
                member, member.c.family_id == unstructured_attribute_family.c.id
            )
        )
        .where(
            unstructured_attribute_family.c.id.in_(
                select(member.c.family_id).where(
                    member.c.attribute_id.in_(attribute_ids)
                )
            )
        )
    )
    return [dict(row) for row in conn.execute(statement).mappings()]


def all_family_rows(conn: Connection) -> list[dict[str, Any]]:
    """Every family and its members, for a caller that wants the whole set.

    :func:`family_rows` answers "what do these attributes belong to", which is
    the query-time question. This answers "what does the corpus state more than
    once", which is the one asked when surveying the ontology rather than
    using it.
    """
    member = unstructured_attribute_family__attribute
    statement = select(
        unstructured_attribute_family.c.id.label("family_id"),
        unstructured_attribute_family.c.canonical,
        unstructured_attribute_family.c.dimension,
        unstructured_attribute_family.c.note,
        member.c.attribute_id,
        member.c.qualifier,
        member.c.value,
    ).select_from(
        unstructured_attribute_family.join(
            member, member.c.family_id == unstructured_attribute_family.c.id
        )
    )
    return [dict(row) for row in conn.execute(statement).mappings()]


def relation_rows(conn: Connection, attribute_ids: list[str]) -> list[dict[str, Any]]:
    """Every adjudicated relation holding *between* two of *attribute_ids*.

    Both sides restricted to the set, which is what keeps this bounded: a
    single attribute can carry dozens of relations across the corpus, but only
    the ones between sources in front of the model can change how it reads
    them.
    """
    if not attribute_ids:
        return []
    relation = unstructured_attribute__attribute
    left = unstructured_attribute.alias("a")
    right = unstructured_attribute.alias("b")
    statement = (
        select(
            relation.c.attribute_id_a,
            relation.c.attribute_id_b,
            relation.c.relation,
            relation.c.canonical,
            relation.c.reason,
            relation.c.qualifier_a,
            relation.c.qualifier_b,
            relation.c.aggregate_side,
            left.c.name.label("name_a"),
            right.c.name.label("name_b"),
        )
        .select_from(
            relation.join(left, left.c.id == relation.c.attribute_id_a).join(
                right, right.c.id == relation.c.attribute_id_b
            )
        )
        .where(
            relation.c.attribute_id_a.in_(attribute_ids),
            relation.c.attribute_id_b.in_(attribute_ids),
        )
    )
    return [dict(row) for row in conn.execute(statement).mappings()]


def index_families(rows: list[dict[str, Any]]) -> dict:
    """Family rows as ``{"by_attribute": ..., "families": ...}``."""
    families: dict[str, dict] = {}
    by_attribute: dict[str, str] = {}
    for row in rows:
        family = families.setdefault(
            str(row["family_id"]),
            {
                "canonical": row["canonical"],
                "dimension": row["dimension"],
                "note": row["note"],
                "members": [],
            },
        )
        family["members"].append(
            {
                "attribute_id": str(row["attribute_id"]),
                "qualifier": row["qualifier"],
                "value": row["value"],
            }
        )
        by_attribute[str(row["attribute_id"])] = str(row["family_id"])
    return {"by_attribute": by_attribute, "families": families}


def relation_lines(relations: list[dict[str, Any]]) -> list[str]:
    """The relations between the sources, grouped by what they warn about."""
    by_relation: dict[str, list[dict]] = {}
    for row in relations:
        by_relation.setdefault(row["relation"], []).append(row)

    lines: list[str] = []
    for relation, heading in RELATION_HEADINGS:
        rows = by_relation.get(relation)
        if not rows:
            continue
        lines.append(heading)
        for row in rows:
            left, right = row["name_a"], row["name_b"]
            if relation == "aggregate_and_part" and row["aggregate_side"] == "b":
                left, right = right, left
            # The adjudicator's reason is already one sentence naming the
            # distinction, which is more use than restating the relation name.
            detail = row["reason"] or row["canonical"] or ""
            lines.append(f"  - '{left}' and '{right}': {detail}")
    return lines


def note_text(families: dict, used: list[str], relations: list[dict[str, Any]]) -> str:
    """What the ontology knows about a set of sources, as prompt text.

    The families name rival readings of one indicator, which is this pipeline's
    costliest silent failure: every figure is genuine, so nothing in the chunk
    text signals the conflict. The relations guard the adjacent mistake —
    reading a component as a total, or a level as a growth rate — between
    sources that happen to arrive together.
    """
    lines: list[str] = []
    for family_id in used:
        family = families.get(family_id) or {}
        if not family.get("note") or family.get("dimension") not in RIVAL_DIMENSIONS:
            continue
        lines.append(
            f"'{family['canonical']}' is reported more than once in this corpus:"
        )
        lines += [
            f"  - {m['qualifier']}: {m['value']}"
            for m in family["members"]
            if m.get("qualifier") and m.get("value")
        ]
        lines.append(f"  {family['note']}")

    lines_from_relations = relation_lines(relations)
    if lines_from_relations:
        if lines:
            lines.append("")
        lines += lines_from_relations

    if not lines:
        return ""
    return "How the sources below relate:\n" + "\n".join(lines)


def ontology_note(attribute_ids: list[str], conn: Connection | None = None) -> str:
    """Everything the ontology knows about *attribute_ids*, as prompt text.

    Empty when there is nothing to warn about, which is the common case and is
    meant to read as silence rather than as a heading with nothing under it.
    """
    if not attribute_ids:
        return ""
    if conn is None:
        from auto_ontology.dal.session import get_engine

        with get_engine().connect() as owned:
            return ontology_note(attribute_ids, owned)

    index = index_families(family_rows(conn, attribute_ids))
    relations = relation_rows(conn, attribute_ids)
    used = list(
        dict.fromkeys(
            index["by_attribute"][i]
            for i in attribute_ids
            if i in index["by_attribute"]
        )
    )
    return note_text(index["families"], used, relations)
