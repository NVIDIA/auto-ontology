# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Relate: how the extracted attributes stand to one another.

Extraction and linking give each attribute its own evidence. This asks what
holds *between* attributes, which is what makes the set an ontology rather than
a list, and it is the stage ``ontology.py`` reads at query time.

Four steps, in one pass:

1. **Propose.** Attribute vectors within a distance of each other are candidate
   pairs. Cheap, and wrong on its own: "EU employment rate" and "EU
   unemployment rate" sit 0.17 apart and mean opposite things.
2. **Adjudicate.** An LLM says what the relation actually is, and why, so the
   edge carries provenance. The rejections are kept — ``unrelated`` and
   ``inverse`` are the expensive part of the answer, and storing them is what
   stops a later pass re-proposing pairs already dismissed.
3. **Group.** The same-measure verdicts are connected components over the
   attributes: a family is one quantity the corpus states more than once.
   ``aggregate_and_part`` is deliberately excluded — an EU total and a country
   value are both correct and do not compete, so presenting them as rival
   readings would invent an ambiguity.
4. **Label.** Each family member gets the qualifier that identifies it — the
   place, the period or the publication — and the figure it asserts, plus one
   sentence a reader needs so they do not mistake one member's figure for
   another's.

Step 4 is the point of the whole stage. The costliest failure on this corpus is
answering from one of several true readings of an indicator without knowing the
others exist: one question scored zero for quoting 58.9%/18.4% from ESDE 2024
where the reference used 65.7%/25.1% from the Joint Employment Report. Both
figures are real. Nothing in either passage says the other exists.

Usage
-----
    python -m unstructured_ao.relate --dry-run
    python -m unstructured_ao.relate --limit 20
    python -m unstructured_ao.relate
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from sqlalchemy import Connection, text

import unstructured_ao.config as config  # noqa: F401  — Ray override
from auto_ontology.utils.llm_invoke import (
    get_llm_client,
    limit_inflight,
    safe_invoke_with_structured_output,
)
from auto_ontology.utils.model_config import resolve

logger = logging.getLogger(__name__)

#: How far apart two attribute vectors may be and still be worth asking about.
#: 0.20 cosine distance: wide enough that the near-misses show up in the
#: verdicts as ``unrelated`` rather than being silently excluded, which is how
#: the threshold was checked.
CANDIDATE_DISTANCE = 0.20

#: Adjudications in flight. The judgement is the slow part, not the database.
MAX_INFLIGHT = 8

#: What a pair can be, and how each reads to the model deciding. Matches the
#: check constraint on ``unstructured_attribute__attribute``.
RELATIONS = {
    "same_measure_different_place": (
        "the same measure reported for different countries or regions"
    ),
    "same_measure_different_period": (
        "the same measure reported for different years, quarters or forecasts"
    ),
    "same_measure_different_source": (
        "the same measure for the same place and period, but from different "
        "surveys or publications, so the values may disagree"
    ),
    "aggregate_and_part": (
        "one is an aggregate (e.g. the EU total) and the other a component of it"
    ),
    "related_but_distinct": "genuinely different measures that are topically adjacent",
    "inverse": (
        "measures that move oppositely by construction, such as employment and "
        "unemployment"
    ),
    "unrelated": "no useful relation",
}

#: The relations that make two attributes readings of one measure, and so
#: members of one family.
FAMILY_RELATIONS = {
    "same_measure_different_place",
    "same_measure_different_source",
    "same_measure_different_period",
}

#: Which dimensions a family may vary along, per the check constraint.
DIMENSIONS = ("place", "period", "source")

RelationName = Literal[
    "same_measure_different_place",
    "same_measure_different_period",
    "same_measure_different_source",
    "aggregate_and_part",
    "related_but_distinct",
    "inverse",
    "unrelated",
]


class PairVerdict(BaseModel):
    """What one LLM call says about one candidate pair."""

    relation: RelationName
    reason: str | None = Field(None, description="One short sentence.")
    canonical: str | None = Field(
        None, description="The shared measure in 2-6 words, or null if unrelated."
    )
    aggregate: Literal["A", "B"] | None = Field(
        None, description="Which side is the whole. Only for aggregate_and_part."
    )
    qualifier_a: str | None = Field(
        None, description="What distinguishes A: place, period or source, 1-4 words."
    )
    qualifier_b: str | None = Field(
        None, description="What distinguishes B, 1-4 words."
    )


class FamilyMemberLabel(BaseModel):
    id: str
    qualifier: str | None = Field(
        None, description="What identifies this reading: country, year or publication."
    )
    value: str | None = Field(
        None, description="The figure(s) it asserts with units, or null."
    )


class FamilyLabel(BaseModel):
    """What one LLM call says about one family."""

    canonical: str = Field(description="The shared measure, 2-8 words.")
    dimension: Literal["place", "period", "source"] | None = Field(
        None, description="What varies across members."
    )
    members: list[FamilyMemberLabel] = Field(default_factory=list)
    note: str | None = Field(
        None,
        description=(
            "One sentence a reader needs so they do not mistake one member's "
            "figure for another's, or null if the members do not conflict."
        ),
    )


_PAIR_SYSTEM = (
    "You classify the relation between two statistical indicators extracted "
    "from EU labour-market reports."
)

_PAIR_TEMPLATE = """Indicator A
  name: {a_name}
  description: {a_description}
  document: {a_document}

Indicator B
  name: {b_name}
  description: {b_description}
  document: {b_document}

Choose exactly one relation from:
{options}
"""

_FAMILY_SYSTEM = (
    "You label statistical indicators extracted from EU labour-market reports."
)

_FAMILY_TEMPLATE = """These indicators were judged to measure the same thing.

{members}

Give the shared measure, what varies across the members, a qualifier and value
for each, and the one sentence a reader needs to tell them apart.
"""

#: Candidate pairs, by distance between the attribute embeddings. Reads the
#: vector table directly: the store's search interface answers "nearest to this
#: query", and what is wanted here is every pair within a radius of each other.
_CANDIDATES_SQL = """
SELECT a.langchain_id::text AS a_id,
       b.langchain_id::text AS b_id,
       (a.embedding <=> b.embedding) AS distance
  FROM {schema}.{table} a
  JOIN {schema}.{table} b ON a.langchain_id < b.langchain_id
 WHERE a.embedding <=> b.embedding <= :threshold
 ORDER BY a.embedding <=> b.embedding
"""

_ATTRIBUTES_SQL = """
SELECT a.id, a.name, a.description, d.name AS document
  FROM unstructured_attribute a
  JOIN unstructured_term t ON t.id = a.term_id
  JOIN unstructured_document d ON d.id = t.document_id
"""

#: Every same-measure edge already adjudicated, whenever and by whomever.
_FAMILY_EDGES_SQL = """
SELECT attribute_id_a AS a_id, attribute_id_b AS b_id, relation
  FROM unstructured_attribute__attribute
 WHERE relation = ANY(:relations)
"""

_PAIR_UPSERT = """
INSERT INTO unstructured_attribute__attribute
       (attribute_id_a, attribute_id_b, relation, canonical, reason,
        qualifier_a, qualifier_b, aggregate_side, distance, model)
VALUES (:a, :b, :relation, :canonical, :reason,
        :qualifier_a, :qualifier_b, :aggregate_side, :distance, :model)
ON CONFLICT (attribute_id_a, attribute_id_b) DO UPDATE SET
    relation = EXCLUDED.relation,
    canonical = EXCLUDED.canonical,
    reason = EXCLUDED.reason,
    qualifier_a = EXCLUDED.qualifier_a,
    qualifier_b = EXCLUDED.qualifier_b,
    aggregate_side = EXCLUDED.aggregate_side,
    distance = EXCLUDED.distance,
    model = EXCLUDED.model
"""


def adjudicator_model() -> str:
    """The model that decides, recorded on every edge it decides."""
    return resolve("REASONING", "MODEL")


def candidate_pairs(
    conn: Connection, threshold: float = CANDIDATE_DISTANCE
) -> list[dict[str, Any]]:
    """Every pair of attributes within *threshold* of each other, closest first."""
    statement = _CANDIDATES_SQL.format(
        schema=config.ATTRIBUTE_SCHEMA, table=config.ATTRIBUTE_TABLE
    )
    return [
        dict(row)
        for row in conn.execute(text(statement), {"threshold": threshold}).mappings()
    ]


def attribute_rows(conn: Connection) -> dict[str, dict[str, Any]]:
    """``{attribute id: name, description, document}`` for the whole corpus."""
    return {
        str(row["id"]): dict(row)
        for row in conn.execute(text(_ATTRIBUTES_SQL)).mappings()
    }


def family_edges(conn: Connection) -> list[dict[str, Any]]:
    """Every same-measure edge in the database, however it got there.

    Families are grouped over these rather than over one pass's verdicts. A
    pass adjudicates the pairs its threshold proposes, which is a fraction of
    what has ever been adjudicated; grouping only those and then rebuilding the
    table would drop every family the rest of the edges support.
    """
    return [
        dict(row)
        for row in conn.execute(
            text(_FAMILY_EDGES_SQL), {"relations": sorted(FAMILY_RELATIONS)}
        ).mappings()
    ]


def _adjudicate_one(
    pair: dict[str, Any], attributes: dict[str, dict[str, Any]], llm
) -> dict[str, Any] | None:
    """One pair, one call. The record to keep, or ``None`` if unanswerable."""
    a = attributes.get(pair["a_id"])
    b = attributes.get(pair["b_id"])
    if not a or not b:
        return None

    messages = [
        SystemMessage(content=_PAIR_SYSTEM),
        HumanMessage(
            content=_PAIR_TEMPLATE.format(
                a_name=a["name"],
                a_description=a["description"] or "",
                a_document=a["document"][:60],
                b_name=b["name"],
                b_description=b["description"] or "",
                b_document=b["document"][:60],
                options="\n".join(f"  {k}: {v}" for k, v in RELATIONS.items()),
            )
        ),
    ]
    try:
        verdict = safe_invoke_with_structured_output(llm, messages, PairVerdict)
    except Exception as exc:  # noqa: BLE001 — one pair must not end the pass
        return {"error": f"{type(exc).__name__}: {exc}"[:120]}
    return {
        "a_id": pair["a_id"],
        "b_id": pair["b_id"],
        "a_name": a["name"],
        "b_name": b["name"],
        "distance": float(pair["distance"]),
        **verdict.model_dump(),
    }


def adjudicate(
    pairs: list[dict[str, Any]],
    attributes: dict[str, dict[str, Any]],
    *,
    progress: bool = True,
) -> list[dict[str, Any]]:
    """What relation holds for each candidate pair, rejections included."""
    llm = get_llm_client(max_tokens=1024)
    verdicts: list[dict[str, Any]] = []
    counts: dict[str, int] = collections.Counter()

    with (
        limit_inflight(MAX_INFLIGHT),
        ThreadPoolExecutor(max_workers=MAX_INFLIGHT) as pool,
    ):
        futures = [
            pool.submit(_adjudicate_one, pair, attributes, llm) for pair in pairs
        ]
        for done, future in enumerate(as_completed(futures), 1):
            record = future.result()
            if record is None:
                continue
            if "error" in record:
                if progress:
                    print(f"  [{done}/{len(pairs)}] {record['error']}", flush=True)
                continue
            verdicts.append(record)
            counts[record["relation"]] += 1
            if progress:
                print(
                    f"  [{done}/{len(pairs)}] d={record['distance']:.3f}  "
                    f"{record['relation']:30} {record['a_name'][:32]} | "
                    f"{record['b_name'][:32]}",
                    flush=True,
                )

    if progress:
        print("\nrelation counts:")
        for relation, count in counts.most_common():
            print(f"  {relation:32} {count}")
    return verdicts


def group_families(edges: list[dict[str, Any]]) -> list[list[str]]:
    """Connected components over the same-measure edges, largest first.

    Transitive on purpose: if A and B are the same measure for different years
    and B and C for different places, all three are readings of one quantity
    and a reader shown two of them needs to know about the third.
    """
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: str, b: str) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_a] = root_b

    for edge in edges:
        if edge["relation"] in FAMILY_RELATIONS:
            union(str(edge["a_id"]), str(edge["b_id"]))

    grouped: dict[str, list[str]] = collections.defaultdict(list)
    for node in parent:
        grouped[find(node)].append(node)
    return sorted(
        (members for members in grouped.values() if len(members) > 1),
        key=len,
        reverse=True,
    )


def _label_one(
    group: list[str], attributes: dict[str, dict[str, Any]], llm
) -> dict[str, Any] | None:
    """One family, one call."""
    known = [i for i in group if i in attributes]
    if not known:
        return None

    members = "\n\n".join(
        f"id: {i}\n  name: {attributes[i]['name']}\n"
        f"  description: {attributes[i]['description'] or ''}\n"
        f"  document: {attributes[i]['document'][:70]}"
        for i in known
    )
    messages = [
        SystemMessage(content=_FAMILY_SYSTEM),
        HumanMessage(content=_FAMILY_TEMPLATE.format(members=members)),
    ]
    try:
        label = safe_invoke_with_structured_output(llm, messages, FamilyLabel)
    except Exception as exc:  # noqa: BLE001 — one family must not end the pass
        return {"error": f"{type(exc).__name__}: {exc}"[:120]}
    return {**label.model_dump(), "attribute_ids": known}


def label_families(
    groups: list[list[str]],
    attributes: dict[str, dict[str, Any]],
    *,
    progress: bool = True,
) -> tuple[list[dict[str, Any]], int]:
    """The qualifier, value and disambiguating note for each family.

    Returns ``(labelled, failed)``.  *failed* counts groups whose call errored;
    a group with no surviving attributes is dropped without counting, since
    there is nothing to label.
    """
    llm = get_llm_client(max_tokens=2048)
    labelled: list[dict[str, Any]] = []
    failed = 0

    with (
        limit_inflight(MAX_INFLIGHT),
        ThreadPoolExecutor(max_workers=MAX_INFLIGHT) as pool,
    ):
        futures = [pool.submit(_label_one, group, attributes, llm) for group in groups]
        for done, future in enumerate(as_completed(futures), 1):
            label = future.result()
            if label is None:
                continue
            if "error" in label:
                failed += 1
                if progress:
                    print(f"  [{done}/{len(groups)}] {label['error']}", flush=True)
                continue
            labelled.append(label)
            if progress:
                print(
                    f"  [{done}/{len(groups)}] {len(label['attribute_ids'])} members, "
                    f"by {label.get('dimension')}: {label.get('canonical')}",
                    flush=True,
                )
    return labelled, failed


def write_relations(
    conn: Connection, verdicts: list[dict[str, Any]], *, model: str | None = None
) -> dict[str, int]:
    """Upsert one edge per verdict. Returns ``{written, skipped}``.

    Upserted, not replaced: every pass adjudicates only what its threshold
    proposed, so clearing the table first would discard everything judged by a
    wider pass. An edge re-judged is overwritten, which is the only case where
    a later opinion should win.
    """
    model = model or adjudicator_model()
    known = {
        row[0] for row in conn.execute(text("SELECT id FROM unstructured_attribute"))
    }

    skipped = 0
    for verdict in verdicts:
        a, b = verdict["a_id"], verdict["b_id"]
        if a not in known or b not in known:
            skipped += 1
            continue
        qualifier_a, qualifier_b = (
            verdict.get("qualifier_a"),
            verdict.get("qualifier_b"),
        )
        side = (verdict.get("aggregate") or "").strip().lower() or None
        # The table stores one row per unordered pair, and the adjudicator's A
        # and B follow its prompt, so everything that names a side is remapped
        # rather than assumed to survive the swap.
        if a > b:
            a, b = b, a
            qualifier_a, qualifier_b = qualifier_b, qualifier_a
            side = {"a": "b", "b": "a"}.get(side, side)
        conn.execute(
            text(_PAIR_UPSERT),
            {
                "a": a,
                "b": b,
                "relation": verdict["relation"],
                "canonical": verdict.get("canonical"),
                "reason": verdict.get("reason"),
                "qualifier_a": qualifier_a,
                "qualifier_b": qualifier_b,
                "aggregate_side": (
                    side if verdict["relation"] == "aggregate_and_part" else None
                ),
                "distance": verdict.get("distance"),
                "model": model,
            },
        )
    return {"written": len(verdicts) - skipped, "skipped": skipped}


def write_families(
    conn: Connection, families: list[dict[str, Any]], *, model: str | None = None
) -> int:
    """Replace the family tables with *families*. Returns how many were written.

    Rebuilt wholesale, not patched: membership comes from the edges, so a
    re-adjudication can split or merge a family, and updating rows in place
    would leave one standing that no edge supports. Safe to replace only
    because *families* was grouped over every edge in the database — see
    :func:`family_edges`.
    """
    model = model or adjudicator_model()
    known = {
        row[0] for row in conn.execute(text("SELECT id FROM unstructured_attribute"))
    }

    conn.execute(text("DELETE FROM unstructured_attribute_family"))
    written = 0
    for family in families:
        members = [i for i in family.get("attribute_ids", []) if i in known]
        if len(members) < 2:
            continue
        family_id = str(uuid.uuid4())
        dimension = (family.get("dimension") or "").strip().lower() or None
        conn.execute(
            text(
                "INSERT INTO unstructured_attribute_family "
                "(id, canonical, dimension, note, model) "
                "VALUES (:id, :canonical, :dimension, :note, :model)"
            ),
            {
                "id": family_id,
                "canonical": family.get("canonical") or "(unnamed)",
                "dimension": dimension if dimension in DIMENSIONS else None,
                "note": family.get("note"),
                "model": model,
            },
        )
        labels = {m.get("id"): m for m in family.get("members", [])}
        for attribute_id in members:
            label = labels.get(attribute_id, {})
            conn.execute(
                text(
                    "INSERT INTO unstructured_attribute_family__attribute "
                    "(family_id, attribute_id, qualifier, value) "
                    "VALUES (:family_id, :attribute_id, :qualifier, :value)"
                ),
                {
                    "family_id": family_id,
                    "attribute_id": attribute_id,
                    "qualifier": label.get("qualifier"),
                    "value": label.get("value"),
                },
            )
        written += 1
    return written


def build(
    *,
    threshold: float = CANDIDATE_DISTANCE,
    limit: int | None = None,
    spread: bool = False,
    dry_run: bool = False,
    out: str | None = None,
    progress: bool = True,
) -> dict[str, int]:
    """The whole stage: propose, adjudicate, group, label, write.

    *dry_run* stops before writing, which is how a threshold or a prompt change
    is reviewed: the verdicts are the expensive part and *out* keeps them.
    """
    from auto_ontology.dal.session import get_engine

    engine = get_engine()
    with engine.connect() as conn:
        pairs = candidate_pairs(conn, threshold)
        attributes = attribute_rows(conn)

    if limit:
        # Spread samples the distance range instead of taking the closest, which
        # is what shows whether the threshold is in the right place: the far end
        # should be mostly rejections.
        step = max(1, len(pairs) // limit) if spread else 1
        pairs = pairs[::step][:limit]

    if progress:
        print(
            f"{len(pairs)} candidate pair(s) at distance <= {threshold}\n"
            f"adjudicator: {adjudicator_model()}\n",
            flush=True,
        )

    verdicts = adjudicate(pairs, attributes, progress=progress)

    # The edges go in before the families are grouped, so the grouping sees
    # every edge ever adjudicated and not just this pass's. On a dry run
    # nothing is written, so the two are combined in memory instead to preview
    # the same answer.
    relations = {"written": 0, "skipped": 0}
    if not dry_run:
        with engine.begin() as conn:
            relations = write_relations(conn, verdicts)

    with engine.connect() as conn:
        edges = family_edges(conn)
    if dry_run:
        edges = edges + [v for v in verdicts if v["relation"] in FAMILY_RELATIONS]

    groups = group_families(edges)
    if progress:
        print(f"\n{len(groups)} family/families to label", flush=True)
    families, failed_labels = label_families(groups, attributes, progress=progress)

    if out:
        from pathlib import Path

        Path(out).write_text(
            json.dumps({"verdicts": verdicts, "families": families}, indent=2)
        )
        if progress:
            print(f"\nwrote {out}", flush=True)

    if dry_run:
        if progress:
            print(
                f"\ndry run: {len(verdicts)} verdict(s) and {len(families)} "
                "family/families, nothing written",
                flush=True,
            )
        return {"relations": 0, "skipped": 0, "families": 0}

    # write_families replaces every stored family, so a partial labelling would
    # silently delete the families whose call failed.
    if failed_labels:
        logger.error(
            "Labelling failed for %d of %d families; kept the stored families.",
            failed_labels,
            len(groups),
        )
        written_families = 0
    else:
        with engine.begin() as conn:
            written_families = write_families(conn, families)
    if progress:
        print(
            f"\nwrote {relations['written']} relation(s) and {written_families} "
            f"family/families; skipped {relations['skipped']} pair(s) whose "
            "attributes no longer exist",
            flush=True,
        )
    return {
        "relations": relations["written"],
        "skipped": relations["skipped"],
        "families": written_families,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=CANDIDATE_DISTANCE)
    parser.add_argument("--limit", type=int, help="Adjudicate only the first N pairs.")
    parser.add_argument(
        "--spread",
        action="store_true",
        help="Sample across the distance range instead of taking the closest.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Adjudicate and report without writing."
    )
    parser.add_argument("--out", help="Also write the verdicts to this JSON file.")
    args = parser.parse_args()

    build(
        threshold=args.threshold,
        limit=args.limit,
        spread=args.spread,
        dry_run=args.dry_run,
        out=args.out,
    )


if __name__ == "__main__":
    main()
