# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-slot derivations for the SQL candidate pool.

A candidate slot varies along two independent axes. The *strategy* changes how
the candidate reasons its way to SQL; the *schema reading* changes which tables
and columns it reaches for. Both are delivered as extra system messages on top
of the one shared SQL prompt, so nothing here duplicates the prompt itself.

The strategies below are two-stage on purpose. Merely asking for a richer
``thought`` field in the single SQL call does not create an independent
reasoning path — the model writes the same query and narrates it differently.
Producing the plan or decomposition as its own structured artifact, in its own
LLM call, is what makes the second call arrive somewhere else.
"""

import logging
import random
from typing import Any

from auto_ontology.retrieval.text_to_sql import candidate_flags as flags

logger = logging.getLogger(__name__)


# Round-robin order for slots 1..N-1. Slot 0 is always baseline so that
# BIRD_NCAND=1 reproduces single-candidate behaviour exactly.
STRATEGY_TAGS = (
    "baseline",
    "query_plan",
    "decomposition",
    "synthetic_examples",
    "query_plan_b",
    "alt_table_set",
)

# Schema readings, orthogonal to the strategies above. The strategies vary how
# a candidate reasons; none of them vary which tables it reads, and that is
# where pooled candidates agree most. Slot 0 stays free, so enabling this can
# only add readings and never removes the one we already get.
SCHEMA_SLOT_ROLES = ("free", "join_preferring", "alternative_binding", "free")

JOIN_PREFERRING = (
    "SCHEMA READING FOR THIS CANDIDATE — prefer the joined reading.\n"
    "Other candidates are writing the single-table reading of this question, so "
    "do not duplicate it. When an attribute the question needs exists BOTH in a "
    "table you already have AND in another table reachable by a foreign key, use "
    "the reachable table's column and add the join. Where two readings are both "
    "defensible, take the one that touches more tables.\n"
    "This is a reading to explore, not a rule to force: if joining would change "
    "what the question asks for, or no second table carries the attribute, write "
    "the query you believe is correct."
)

ALTERNATIVE_BINDING = (
    "SCHEMA READING FOR THIS CANDIDATE — take the second-choice binding.\n"
    "Another candidate is already writing the most obvious reading, so yours must "
    "explore a different one. For each ambiguous term below: name the column you "
    "would reach for first, then commit to a DIFFERENT plausible column and write "
    "the query under that reading, adding whatever join it needs.\n"
    "If a term genuinely has only one plausible binding, leave it alone — the "
    "point is to cover a reading the pool would otherwise miss, not to be wrong "
    "on purpose."
)

ALT_TABLE_SET_STRATEGY = (
    "SCHEMA DIVERGENCE FOR THIS CANDIDATE — change the base table set.\n"
    "Other candidates will take the most obvious fact table for this question. "
    "You must answer using a DIFFERENT primary table when a foreign-key-reachable "
    "alternative carries the same attribute (or a clearer one). Prefer a join "
    "path the obvious reading skips. Name the table you are deliberately not "
    "using as the sole base, and the table you use instead.\n"
    "If no second table is defensible, write the best query you can — but when "
    "two table sets are both plausible, you MUST take the less obvious one.\n"
    "Raw SQL only in the final answer."
)

QUERY_PLAN_ARTIFACT_PROMPT = (
    "STAGE 1 OF 2 — QUERY PLAN ONLY. Do not write SQL. Build a concrete "
    "numbered execution plan for the target question from the supplied schema "
    "and evidence. Name every table and join key, exact filters/literals, "
    "target row grain, aggregation, projection, ordering, and LIMIT. Return "
    "only the structured plan."
)

DECOMPOSITION_ARTIFACT_PROMPT = (
    "STAGE 1 OF 2 — DECOMPOSITION ONLY. Do not write the final SQL. Break the "
    "target question into 2-4 independently answerable sub-questions. For "
    "each, identify the exact tables, columns, joins, filters, and aggregate "
    "needed, then state how their answers compose at the requested row grain. "
    "Return only the structured decomposition."
)

DECOMPOSITION_TREE_ARTIFACT_PROMPT = (
    "STAGE 1 OF 2 — DIVIDE AND CONQUER. Do not write the final SQL.\n\n"
    "Start from the target question. State what it asks for, then write "
    "top-level pseudo SQL in which every part you cannot yet resolve is left "
    "as bracketed natural language, e.g.\n"
    "  SELECT T1.gender FROM client AS T1 "
    "WHERE <youngest client in the lowest average salary branch>\n\n"
    "Then solve each bracketed part as its own numbered sub-question with its "
    "own analysis and its own pseudo SQL. If a sub-question still contains a "
    "bracketed part, give it a nested label (1.1, 1.2) and solve that too. "
    "Stop only when no brackets remain.\n\n"
    "Finish with two separate steps: assembly, substituting each node's "
    "pseudo SQL into its parent's placeholder from the bottom up; and "
    "simplification, collapsing nested subqueries into joins and dropping "
    "redundant clauses without changing the row grain.\n\n"
    "Return only the structured decomposition."
)

QUERY_PLAN_TRANSLATION_PROMPT = (
    "STAGE 2 OF 2 — TRANSLATE THE PLAN TO SQL. The plan below is the "
    "intermediate reasoning artifact. Implement it faithfully, but correct any "
    "step that contradicts the visible schema or evidence. Do not add a SQL "
    "clause unless a plan step requires it.\n\nQUERY PLAN:\n{artifact}"
)

DECOMPOSITION_TRANSLATION_PROMPT = (
    "STAGE 2 OF 2 — ASSEMBLE THE FINAL SQL. Solve each decomposition item and "
    "compose them into one executable query at the requested row grain. Drop "
    "any clause that answers no sub-question.\n\nDECOMPOSITION:\n{artifact}"
)

DECOMPOSITION_TREE_TRANSLATION_PROMPT = (
    "STAGE 2 OF 2 — EMIT THE SIMPLIFIED QUERY. The decomposition below already "
    "carries the assembly and simplification steps. Produce the query they "
    "arrive at: every bracketed placeholder resolved, the simplification "
    "applied, and no clause that answers no sub-question. Correct any step "
    "that contradicts the visible schema or evidence, and keep the row grain "
    "the main analysis states.\n\nDECOMPOSITION:\n{artifact}"
)

SYNTHETIC_USE_PROMPT = (
    "STAGE 2 OF 2 — SOLVE THE TARGET QUESTION. The Reference Query Patterns "
    "section does not hold precedent from other databases this time: it holds "
    "newly generated and execution-checked demonstrations from this exact "
    "target schema. Use their schema-specific joins and conventions when "
    "relevant, but answer the target question rather than copying an example."
)


def synthetic_artifact_prompt() -> str:
    """Stage-1 instruction for inventing same-schema demonstrations.

    Built per call because the example count is an environment flag. When
    reasoning is enabled each example also carries its derivation: a bare
    question/answer pair shows a question leaping straight to finished SQL,
    which is the guessing behaviour the rule block then tries to suppress.
    """
    base = (
        "STAGE 1 OF 2 — ONLINE SAME-SCHEMA DEMONSTRATIONS ONLY. Generate "
        f"{flags.synthetic_n()} realistic question-to-SQL examples using ONLY "
        "the supplied target schema and documented join keys. Make them "
        "structurally useful for the target question (relevant joins, filters, "
        "aggregation, or output grain) but do not paraphrase or solve the target "
        "question. Vary them: different join depths, different aggregations, and "
        "both filtered and unfiltered shapes, so together they show how this "
        "schema is meant to be queried. Each SQL must be complete and executable. "
        "Return only the structured examples"
    )
    if not flags.synthetic_reasoning():
        return base + "."
    return base + (
        " with their derivations.\n\nFor each example also give the reasoning "
        "that produces the SQL: the tables needed and why, the join keys, the "
        "filters with exact literals, the row grain, and the projection. Write it "
        "as the derivation a reader could follow to rebuild the query, not as a "
        "description of the finished query."
    )


def strategy_for_slot(index: int, n_candidates: int) -> str:
    """Which strategy slot *index* runs.

    ``BIRD_PIN_STRATEGY`` wins over ``BIRD_SLOT_PLAN``, which wins over the
    round-robin. A pin also makes slot 0 sample; otherwise one draw of N would
    be deterministic and understate the spread the pin is meant to measure.
    """
    if n_candidates < 2:
        return "baseline"

    pinned = flags.pin_strategy()
    if pinned:
        if pinned in STRATEGY_TAGS:
            return pinned
        logger.warning(
            "BIRD_PIN_STRATEGY=%r is not one of %s; ignoring",
            pinned,
            ", ".join(STRATEGY_TAGS),
        )

    plan = flags.slot_plan()
    if plan:
        unknown = sorted({tag for tag in plan if tag not in STRATEGY_TAGS})
        if unknown:
            logger.warning(
                "BIRD_SLOT_PLAN names unknown strategies %s; ignoring the plan",
                ", ".join(unknown),
            )
        else:
            return plan[index % len(plan)]

    if index == 0:
        return "baseline"
    return STRATEGY_TAGS[index % len(STRATEGY_TAGS)]


def slot_samples(index: int, n_candidates: int) -> bool:
    """Whether slot *index* runs on the sampling client rather than the base one."""
    if n_candidates < 2:
        return False
    return index > 0 or bool(flags.pin_strategy())


def schema_directive(index: int, entity_columns: list[dict] | None) -> str:
    """The schema-reading directive for slot *index*, or ``""`` for a free slot.

    Ambiguous terms are named inline for the alternative-binding slot. Without
    something concrete to bind to the instruction reads as a vague suggestion
    and the model reverts to the obvious choice, so an unnamed slot falls back
    to the join reading instead.
    """
    if not flags.schema_slots_enabled() or index <= 0:
        return ""
    role = SCHEMA_SLOT_ROLES[index % len(SCHEMA_SLOT_ROLES)]
    if role == "join_preferring":
        return JOIN_PREFERRING
    if role != "alternative_binding":
        return ""
    terms = [
        str(group.get("entity") or "").strip()
        for group in (entity_columns or [])
        if str(group.get("entity") or "").strip()
        and len(group.get("columns") or []) >= 2
    ]
    if not terms:
        return JOIN_PREFERRING
    return (
        ALTERNATIVE_BINDING
        + "\nAmbiguous terms: "
        + ", ".join(f'"{term}"' for term in terms[:6])
    )


def schema_role(index: int) -> str:
    """Name of the schema reading slot *index* was given, for logging."""
    return SCHEMA_SLOT_ROLES[index % len(SCHEMA_SLOT_ROLES)]


def shuffled_tables(tables: list[dict], rng: random.Random) -> list[dict]:
    """Copy *tables* with table order and per-table column order shuffled.

    Schema-order perturbation diversifies candidates even when the model is
    near-deterministic, which matters for providers that ignore an explicit
    temperature. Each table dict is shallow-copied so the caller's list and
    the shared column lists are left untouched.
    """
    out = [dict(table) for table in tables]
    rng.shuffle(out)
    for table in out:
        columns = table.get("columns")
        if isinstance(columns, list):
            columns = list(columns)
            rng.shuffle(columns)
            table["columns"] = columns
    return out


def rotate_examples(examples: list[Any], index: int) -> list[Any]:
    """A deterministic, complementary slice of *examples* for slot *index*.

    Slot 0 keeps every example, preserving the baseline path. Other slots take
    four-example patterns that overlap as little as possible, so candidates do
    not all anchor on the same demonstrations.
    """
    if index == 0 or len(examples) <= 4:
        return list(examples)

    patterns = (
        (0, 2, 4, 6),
        (1, 3, 5, 7),
        (0, 1, 4, 5),
        (2, 3, 6, 7),
    )
    pattern = patterns[(index - 1) % len(patterns)]
    return [examples[i] for i in pattern if i < len(examples)]


def format_decomposition_tree(artifact: Any) -> str:
    """Render a decomposition tree in solve order, deepest nodes first.

    Sorting by label depth puts a child ahead of its parent, so the text is
    already in substitution order by the time stage 2 reads it.
    """

    def _depth(label: str) -> tuple:
        parts = [part for part in str(label).split(".") if part.strip()]
        return (-len(parts), [int(p) if p.isdigit() else 0 for p in parts])

    lines = [
        f"Main analysis: {artifact.main_analysis}",
        f"Main pseudo SQL: {artifact.main_pseudo_sql}",
        "",
    ]
    for node in sorted(artifact.nodes, key=lambda n: _depth(n.label)):
        lines.append(f"Sub-question {node.label}: {node.question}")
        lines.append(f"  Analysis: {node.analysis}")
        lines.append(f"  Pseudo SQL: {node.pseudo_sql}")
    lines.append("")
    lines.append(f"Assembly: {artifact.assembly}")
    lines.append(f"Simplification: {artifact.simplification}")
    return "\n".join(lines)


def format_decomposition_list(artifact: Any) -> str:
    """Render the flat decomposition artifact for the stage-2 prompt."""
    text = "\n".join(f"{i}. {item}" for i, item in enumerate(artifact.sub_questions, 1))
    return text + f"\nComposition: {artifact.composition}"
