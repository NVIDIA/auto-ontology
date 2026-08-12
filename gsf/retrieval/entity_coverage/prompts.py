# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompts for the entity-coverage question_extraction node."""


def create_question_extraction_prompt(question: str) -> str:
    """Single prompt: sanitize the question and extract entity noun phrases."""
    return f"""You rewrite conversational user requests into concise, SQL-ready questions \
AND extract database entity noun phrases from the sanitized intent.

## Part 1 — sanitized_question

Rules:
- Remove personal background, narrative fluff, filler, politeness, and generic request
  framing such as "please", "can you", "show me", "find", or
  "get semantic objects related to". Keep only the underlying domain intent.
- Preserve every factual constraint: numbers, product names, brands, categories, and \
qualifiers such as "similar", "natural ingredients", or "expensive is okay".
- Do NOT invent constraints that are not in the original text.
- If the input is already a direct domain question without generic request framing,
  return it unchanged.
- Output one concise question or search intent, not a paragraph.

Examples:

Input: We're planning a road trip next summer and my whole family loves hiking.
I need a tent that can fit 4 people, and lighter is better since we'll carry it.
sanitized_question: Find a 4-person tent, prioritizing lighter weight.

Input: How many shipments were delivered last month?
sanitized_question: How many shipments were delivered last month?

Input: get semantic objects related to customers or Q1 revenue
sanitized_question: customers or Q1 revenue

## Part 2 — required_entity_name

Populate "required_entity_name" with 1–5 noun phrases that correspond to database \
tables, columns, or relationships. Extract from the sanitized intent.

Preserve the exact casing of terms as they appear in the question. Do not lowercase,
uppercase, or normalize them.

Guidelines for what to include in required_entity_name:
- Subject nouns and domain terms ("invoice", "customer", "shipment")
- Qualified entity phrases that combine a subject with its relevant action or attribute
  ("order shipment", "employee hire", "ticket resolution")
- Filter-item rule: when several words together describe a single item the user wants to
  filter or search for, keep them in one phrase. Do not split modifier, noun, and purpose
  of the same filter item into separate entries.
  Example: "waterproof hiking tent for family camping" → \
["waterproof hiking tent for family camping"],
  not ["waterproof hiking tent", "family camping"].
- Keep names and descriptive text that identify something: brand names, product names,
  vendor names, categories, and other named constants (e.g. "Salomon Speedcross").
- For interrogative words (who/what/which/whose), resolve to the implied entity type
  AND, if the question contains a qualifying descriptor, include it twice: once alone
  and once combined with the resolved type.
  Example: "who are the active assignees" → ["assignee", "active assignee"]

Guidelines for what to exclude from required_entity_name:
- Bare action verbs ("submitted", "approved", "closed", "assigned")
- Numeric values: counts, amounts, prices, years, and other number literals
- Date/time values when they are numeric or calendar literals
- Aggregation indicators ("count", "total", "average", "sum", "min", "max")
  when standing alone
- Status and filter adjectives when standing alone ("open", "active", "high-priority")
- Bare schema-generic words with no domain meaning on their own: "id", "name",
  "type", "code", "key", "value", "description", "label", "title", "flag",
  "uuid", "pk", "fk". Do not emit these as standalone entities.

Date rule: When a question references a time-qualified event, collapse subject + action
+ granularity into one compact phrase ending with "date".
  Example: "invoices closed in Q2" → ["invoice", "invoice closure quarter date"]
  Example: "orders placed last year" → ["order", "order placement date"]

Examples:
  Q: "How many shipments were delivered last month?"
  → required_entity_name: ["shipment", "shipment delivery month date"]

  Q: "Find a waterproof hiking tent for family camping."
  → required_entity_name: ["waterproof hiking tent for family camping"]

## Input

{question}
"""
