# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompts for the entity-coverage question_extraction node."""

from __future__ import annotations

import os

# Maximum number of entity noun phrases to extract per question during clarification.
# Raise via CLARIFICATION_MAX_ENTITIES env var for complex multi-metric queries.
# Falls back to SQL_GEN_MAX_ENTITIES (then 5) when unset.
_MAX_ENTITIES: int = int(
    os.environ.get(
        "CLARIFICATION_MAX_ENTITIES",
        os.environ.get("SQL_GEN_MAX_ENTITIES", "5"),
    )
)


def format_glossary_section(glossary: list[dict[str, str]] | None) -> str:
    """Render the user-curated Glossary, or "" when there is nothing to inject.

    Entries are a flat list rather than ``rules_to_text``'s ``## name`` headings,
    which would collide with surrounding ``##`` prompt sections.
    """
    entries = [
        f"- {name}: {(entry.get('description') or '').strip()}"
        for entry in glossary or []
        if (name := (entry.get("name") or "").strip())
    ]
    if not entries:
        return ""
    definitions = "\n".join(entries)
    return f"""## Glossary

Definitions the user's organization has registered. Use them to resolve \
abbreviations, shortcuts, and internal jargon in the question.

{definitions}

"""


def format_evidence_section(evidence: str | None) -> str:
    """Render narrowly scoped evidence guidance for question sanitization."""
    if not evidence:
        return ""
    return f"""## Evidence for sanitization

Use this evidence only to disambiguate the domain meaning of the input while producing
`sanitized_question`.

Rules:
- Add at most one concise disambiguating qualifier to the sanitized question, and only
  when it materially improves entity retrieval.
- Do not append or summarize the evidence.
- Do not copy background, explanations, formulas, SQL, values, unrelated terms, or
  inventories of tables or columns into the sanitized question.
- Evidence must not add a second interpretation or broaden the user's request.
- Part 2 must derive entities only from the completed sanitized question, never directly
  from this evidence section.

{evidence}

"""


_SANITIZE_AND_ENTITIES = f"""## Part 1 — sanitized_question

Rules:
- Remove personal background, narrative fluff, filler, politeness, and generic request
  framing such as "please", "can you", "show me", "find", or
  "get semantic objects related to". Keep only the underlying domain intent.
- Preserve every factual constraint: numbers, product names, brands, categories, and \
qualifiers such as "similar", "natural ingredients", or "expensive is okay".
- Do NOT invent constraints that are not in the original text.
- Output one concise question or search intent, not a paragraph.

Examples:

Input: We're planning a road trip next summer and my whole family loves hiking.
I need a tent that can fit 4 people, and lighter is better since we'll carry it.
sanitized_question: Find a 4-person tent, prioritizing lighter weight.

Input: How many shipments were delivered last month?
sanitized_question: How many shipments were delivered last month?

## Part 2 — required_entity_name

Populate "required_entity_name" with 1–{_MAX_ENTITIES} objects. Every object has:
- `type`: `field` for a requested schema concept, or `value` for a literal that can
  be compared with a database field.
- `original_text`: the exact words from the completed sanitized question.
- `meaning`: the schema-oriented concept to use for semantic retrieval.

Extract only from the completed sanitized intent, never directly from Glossary or
Evidence sections. Preserve the exact casing in `original_text`.

Identify complete VALUE spans before FIELD entities:
- Values include names, titles, labels, categories, statuses, alphanumeric codes,
  booleans, numbers, percentages, and dates used as filter constants.
- Keep the complete literal in `original_text`; do not split it.
- Set `meaning` to the kind of field likely to contain the value, not to the literal.
- A word belonging to a value span MUST NOT appear in any other entity. In particular,
  do not also emit a field made from a noun inside a value.

Field entities include requested outputs, metrics, dimensions, subjects, and
relationships. For a field, `meaning` should be a concise schema concept. Resolve
Glossary abbreviations and internal jargon in `meaning`, while `original_text` remains
the words used by the question.

The existing semantic-entity rules still apply to FIELD entities:
- Include subject nouns and domain terms ("invoice", "customer", "shipment").
- Include qualified phrases that combine a subject with its relevant action or
  attribute ("order shipment", "employee hire", "ticket resolution").
- Aggregation-qualified metric rule: when "count", "total", "average", "sum", "min",
  or "max" is attached to a domain noun as the name of a requested metric or column,
  keep it as one field meaning. When the aggregation is only question framing, such as
  "How many shipments", emit the subject field meaning "shipment".
- Filter-item rule: when several words together describe one item the user wants to
  filter or search for, keep them in one complete entity instead of splitting modifier,
  noun, and purpose. For example, keep "waterproof hiking tent for family camping" as
  one field entity.
- For interrogative words (who/what/which/whose), resolve the implied entity type. If
  the question contains a qualifying descriptor, include the base field and the
  qualified field. Example: "who are the active assignees" produces field meanings
  "assignee" and "active assignee".
- Date-field rule: when a time qualifier modifies an event rather than naming a stored
  literal date, collapse subject + action + granularity into a compact field meaning
  ending with "date". Examples: "invoices closed in Q2" includes "invoice closure
  quarter date"; "orders placed last year" includes "order placement date".

Glossary rule: when a question phrase matches a Glossary abbreviation, shortcut, or
internal term, keep the user's phrase in `original_text` and resolve its definition in
the field's `meaning`. Resolution replaces the meaning; it does not add an extra entity.
Example (Glossary: "MRR" = "monthly recurring revenue"):
  "show MRR by region" produces field meanings "monthly recurring revenue" and "region",
  not an additional duplicate "MRR" meaning.

Existing exclusions still apply to FIELD entities:
- Do not emit bare action verbs ("submitted", "approved", "closed", "assigned").
- Do not emit aggregation indicators by themselves.
- Do not emit status/filter adjectives by themselves as fields.
- Do not emit numeric values, date literals, thresholds, names, brands, product names,
  categories, or other named constants as fields. They are now emitted as VALUE
  entities when they are explicit filter constants; this is an extension of the field
  rules, not a replacement for them.
- Exclude schema-generic words with no domain meaning on their own: "id", "name",
  "type", "code", "key", "value", "description", "label", "title", "flag", "uuid",
  "pk", and "fk".

Examples:
  Q: "Did class 2 see The Lion King?"
  → required_entity_name:
    [
      {{"type": "value", "original_text": "class 2", "meaning": "class"}},
      {{"type": "value", "original_text": "The Lion King", "meaning": "movie"}}
    ]
  Do not additionally emit a `class` field: "class" is inside the value "class 2".

  Q: "Which artist illustrated the card Angel of Mercy?"
  → required_entity_name:
    [
      {{"type": "field", "original_text": "artist", "meaning": "card artist"}},
      {{"type": "value", "original_text": "Angel of Mercy", "meaning": "card name"}}
    ]

  Q: "How many shipments were delivered last month?"
  → required_entity_name:
    [
      {{"type": "field", "original_text": "shipments", "meaning": "shipment"}},
      {{"type": "field", "original_text": "delivered last month",
        "meaning": "shipment delivery month date"}}
    ]

  Q: "Find a waterproof hiking tent for family camping."
  → required_entity_name:
    [
      {{"type": "field",
        "original_text": "waterproof hiking tent for family camping",
        "meaning": "waterproof hiking tent for family camping"}}
    ]
"""

_SUBJECT_AND_ACRONYMS = """## Part 3 — subject

Populate "subject" with one short noun phrase naming what the question is about — the
single thing being asked for. Derive it from the sanitized question, resolving any
Glossary entry that applies, and preserve casing the same way Part 2 does.

Exclude from the subject: filters and qualifiers, aggregation words ("count", "total",
"average"), date and time qualifiers, and number literals.

Examples:
  Q: "How many shipments were delivered last month?"
  → subject: "shipment"

  Q: "Find a waterproof hiking tent for family camping."
  → subject: "tent"

  Q: "Which vendors had the highest invoice totals in Q2?"
  → subject: "vendor"

## Part 4 — used_glossary_names

Populate "used_glossary_names" with the names of only the Glossary entries you actually used
to interpret, sanitize, resolve entities in, or determine the subject of this question.
Copy each name exactly as written in the Glossary. Do not infer entries by lexical
matching alone: include an entry only when its definition is semantically relevant.
Return an empty list when no Glossary definition applies.

Example (Glossary contains "MRR: monthly recurring revenue"):
  Q: "Show MRR by region."
  → used_glossary_names: ["MRR"]
"""


def create_question_extraction_prompt(
    question: str,
    glossary: list[dict[str, str]] | None = None,
    *,
    evidence: str | None = None,
    include_subject: bool = True,
) -> str:
    """Sanitize and extract entity noun phrases; optionally also name subject/acronyms.

    Glossary is always injected so the model can resolve abbreviations. Evidence,
    when provided, is restricted to refining the sanitized question before entity
    extraction, including when ``include_subject`` is False.
    """
    glossary_section = format_glossary_section(glossary)
    evidence_section = format_evidence_section(evidence)
    if include_subject:
        intro = (
            "You rewrite conversational user requests into concise, SQL-ready "
            "questions, extract database entity noun phrases from the sanitized "
            "intent, AND name the question's main subject."
        )
        trailing = _SUBJECT_AND_ACRONYMS
    else:
        intro = (
            "You rewrite conversational user requests into concise, SQL-ready "
            "questions and extract database entity noun phrases from the "
            "sanitized intent. Do not produce a subject field or list of used "
            "acronyms."
        )
        trailing = ""

    return f"""{intro}

{_SANITIZE_AND_ENTITIES}
{trailing}
{glossary_section}{evidence_section}## Input

{question}
"""
