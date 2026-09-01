# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text-to-PQL prompt, built in pure Python (no Jinja).

``build_pql_prompt`` assembles the same prompt the ported pipeline used: the
static PQL grammar + "common mistakes" header, then the graph schema, optional
columns/domain-rule/verified-example sections, an optional repair block (the
previous attempt + its error), an optional single-entity explanation block, the
output-format instructions, and the question.
"""

from __future__ import annotations

# Static instructions (PQL grammar + the failure-mode guardrails). Everything
# below the header is assembled conditionally in ``build_pql_prompt``.
_PQL_HEADER = """You are an expert in Kumo's Predictive Query Language (PQL). Translate the user's predictive question
into a single PQL query that runs against the relational graph described below.

## PQL grammar
```
PREDICT <target> [RANK TOP k] FOR EACH <entity_table>.<primary_key> [WHERE <filter>] [ASSUMING <cond>]
PREDICT <target> [RANK TOP k] FOR <entity_table>.<primary_key>=<value> [ASSUMING <cond>]
```
- The TARGET decides the task:
  - Aggregation over a time-stamped event table → regression:
    `AGG(<table>.<column> [WHERE ...], <start>, <end>, <time_unit>)` where AGG ∈ COUNT, SUM, AVG, MIN, MAX.
    The window `(anchor+start, anchor+end]` is relative to the prediction "anchor" time; time_unit ∈
    seconds, minutes, hours, days, weeks, months. Use `*` to count rows: `COUNT(orders.*, 0, 30, days)`.
  - A condition / comparison (`> 0`, `= 0`, `>= 100`, `AND`/`OR`) → binary classification:
    e.g. churn = `COUNT(orders.*, 0, 30, days) = 0`; will-purchase = `COUNT(orders.*, 0, 7, days) > 0`.
  - To predict whether a future event MATCHING A CONDITION occurs, put the condition in a `WHERE` *inside*
    the aggregation, then compare to make it binary. NEVER put a row-level event filter after `FOR EACH`
    (only entity filters go there). Compare a **categorical** column with `=` / `!=` ONLY (never `<`/`>`);
    use `<`/`>`/`<=`/`>=` for **numeric** columns. Examples:
    `COUNT(results.* WHERE results.statusId != 1, 0, 365, days) > 0`  (driver will DNF / not-finish next race)
    `MIN(results.position, 0, 365, days) <= 3`  (finish in the top 3 — numeric position column).
  - A direct categorical/ID column → multi-class; a direct numeric column → regression.
  - `LIST_DISTINCT(<fk_column>, start, end, time_unit) RANK TOP k` → recommendation / link prediction
    (the FK column must be a registered link in the graph).
  - Add `FORECAST n TIMEFRAMES` for multi-step forecasting. FORECAST is **single-entity only** — KumoRFM
    forecasts the time series for exactly ONE entity per query. Name that entity by its PRIMARY KEY with
    `FOR <entity>.<primary_key>=<value>` (never a category, measure, or grouping column). Never frame a
    forecast as "for each" / "per" many entities.
    Choose regression vs FORECAST by the NUMBER OF FUTURE PERIODS asked for, NOT the number of entities:
    - **One future period** ("next month", "next quarter", "over the next N days", an "outlook" / "expected"
      total) → a plain aggregation **regression** `AGG(<event>.<col>, 0, N, <unit>) FOR EACH
      <entity>.<primary_key>` — one expected value per entity for that window. Use this even for a SINGLE
      entity (e.g. "demand for product X next month" → `SUM(product_demand_daily.units, 0, 1, months) FOR
      products.product_id='X'`), and ALWAYS use it when an explanation is wanted.
      NEVER use FORECAST for a single-period question.
    - **Multiple successive periods / a trajectory over time** ("month-by-month", "each of the next 6 months",
      "weekly trend") → `FORECAST n TIMEFRAMES` (single-entity, per the constraint above).
- `FOR EACH <entity>.<pk>` predicts a population. Add `WHERE <entity>.<col> = ...` only for a population
  subset. `FOR <entity>.<pk>=<value>` predicts exactly one entity and MUST be used for a single-entity
  explanation; do not express a single primary key as `FOR EACH ... WHERE <pk> = <value>`.
- Do NOT use `EVALUATE PREDICT ...` — it is unsupported. Emit a plain `PREDICT ...`.

## Common mistakes to avoid (these FAIL validation)
- PQL has NO SQL subqueries. NEVER write `SELECT` anywhere — not in `ASSUMING`, not anywhere else. To reach a
  related table, rely on the graph's foreign keys (the entity→event link is IMPLICIT); to restrict which
  entities are scored, use the SEPARATE entity-selection ```sql block (below), never a subquery inside PREDICT.
- NEVER use SQL time expressions in PQL (`CURRENT_TIMESTAMP`, `NOW()`, `CURRENT_DATE`, `SYSDATE`, …). The
  prediction window is already relative to the data's anchor time via the `(start, end, unit)` args. To scope to
  entities in a future window (e.g. open / at-risk orders), put that date filter in the entity-selection ```sql.
- An aggregation's `WHERE` can ONLY filter columns of the SAME table being aggregated. `COUNT(po_receipts.*
  WHERE po_receipts.status = 'late', …)` is fine; `COUNT(po_receipts.* WHERE purchase_orders.is_critical = ...)`
  is INVALID — you cannot reference another table inside the aggregation. If a needed attribute lives on a
  different table (e.g. is_critical on purchase_orders/components), drop it from the aggregation and filter those
  entities in the entity-selection ```sql block instead, or pick the closest column on the aggregated table.
- `RANK TOP k` is ONLY valid with `LIST_DISTINCT(...)` (link prediction) or a direct multi-categorical column,
  and it goes immediately AFTER the target and BEFORE `FOR EACH` — never after `FOR EACH`. NEVER add RANK to a
  binary (`COUNT(...) > 0` / `= 0`) or numeric-aggregation target. For "most/least likely", emit the binary
  classifier WITHOUT RANK — its probabilities already rank the entities.
- Do NOT predict generic child-event existence/counts for an order-like entity (e.g. `COUNT(order_lines.*) > 0
  FOR EACH sales_orders`, `COUNT(po_receipts.*) FOR EACH purchase_orders`): those relationships are
  near-deterministic and not a useful prediction. Predict an OUTCOME with a status `WHERE` filter (e.g. late /
  slip / short) instead, or answer historical "how many" counts with SQL.
- For recommendation / "which X to recommend / most likely to buy / engage with" questions, use link
  prediction: `LIST_DISTINCT(<event>.<fk>, 0, N, unit) RANK TOP k FOR EACH <entity>.<pk>`, where `<fk>` is a
  registered FK link in the graph (see the verified examples below). RANK TOP goes before FOR EACH.
- Account growth, retention, cross-sell, upsell, and outreach-strategy questions over customer/account
  purchase history are next-best-product recommendations: treat them as the link-prediction case above — rank
  the future product/item FK on the timestamped purchase/order-line event for each customer/account entity,
  when the graph has a customer→product purchase edge (see the verified examples below).
- Spell every identifier exactly as the graph above spells it. Write a name bare when it can be written bare
  (`payments.days_late`); backtick-quote one that cannot, such as a name containing a space — the graph shows
  those already quoted, so copy that form (`people.`Customer ID``). NEVER use double quotes or square brackets.
- `FOR EACH` MUST name the entity by a primary-key column (`FOR EACH customers.customer_id`), never just the
  table. Where the graph marks a table as keyed on several columns TOGETHER (`PRIMARY KEY (`Customer ID`,
  REGION)`), name any ONE of those columns — that identifies the whole key, so do not list them all.
- `FORECAST` is single-entity: forecast ONE entity by its primary key and scope to it with the entity-selection SQL (a single primary-key row). A forecast framed over many entities ("for each" / "per"), or keyed on a non-primary-key column, is wrong.
- Inside an aggregation `WHERE`, qualify the column with its table: `COUNT(payments.* WHERE payments.days_late > 0, 0, 30, days)`.
- Add entity filters ONLY for subset values explicitly named in the user question. Never copy or infer filters
  from verified examples or dataset/domain wording; generic nouns like "business", "customer", "account", or
  "entity" are not filters. If the user asks "which/top/all" entities without a subset value, score the full
  entity population.
- For gain/loss, increase/decrease, or change questions, if the schema exposes an explicit numeric change/delta
  target for the requested metric and horizon, predict that target directly; do not predict the level and
  subtract a baseline in the answer.
- The entity-to-event link is IMPLICIT via the graph's foreign key — NEVER restate the join key inside the
  aggregation. For "orders per APAC customer" write `COUNT(sales_orders.*, 0, 90, days) FOR EACH
  customers.customer_id WHERE customers.region = 'apac'`, NOT
  `COUNT(sales_orders.* WHERE sales_orders.customer_id = customers.customer_id, 0, 90, days)`. Match every
  filter value to what the question actually names and to the column's sampled values (e.g. "key customers"
  is account_tier = 'key', not 'strategic'); never swap in a different tier/region/status than was asked.
- For a slip / late / on-time OUTCOME, classify on the status (or flag) column using its exact values from
  the column table below — e.g. predict order slip with `order_fulfillment.status = 'late'`. (As the
  grammar says, compare a categorical column with `=` / `!=`, never `<` / `>`.)
- For "at risk / will slip / likely in the next N days" questions, scope the entity-selection SQL to
  entities whose due/committed date falls in that future window relative to the latest data — not merely
  `IS NOT NULL`, which ranks stale historical rows."""


_REPAIR_BODY = (
    "Correct the PQL to resolve this error (check table/column/link names and the target shape). Change ONLY what\n"
    "the error requires: preserve every entity-scope filter from the previous attempt (the `FOR EACH ... WHERE`\n"
    "conditions on tier, region, customer, status, etc.) and keep the same aggregation window `(start, end, unit)`,\n"
    "unless that exact predicate or window IS the cause of the error. Fixing one mistake must not silently drop the\n"
    "user's scope or change the time horizon.\n"
    "If the error says entities/indices could not be found or asks to pass `indices`, do NOT add an entity-type\n"
    "filter to make the query run. Keep the user's requested population; the tool resolves entity ids separately."
)

_SNOWFLAKE_CASE_NOTE = (
    "## Identifier casing (Snowflake)\n"
    "This warehouse is Snowflake, where an identifier created WITHOUT quotes is stored UPPERCASE. Write every\n"
    "table AND column name exactly as the Graph section above spells it: uppercase for the bare ones (e.g.\n"
    "`GPU_ALLOCATIONS.*`, not `gpu_allocations.*`; `FOR EACH GPUS.GPU_ID`, not `GPUS.gpu_id`). A name the graph\n"
    "shows backtick-quoted was created quoted and keeps its own mixed case: copy it character for character\n"
    "(`people.`Customer ID``, never `PEOPLE.`CUSTOMER ID``). A mis-cased identifier fails against the graph and\n"
    "the live database."
)

_ENTITY_SQL_INSTRUCTION = (
    "2. To scope WHICH entities to score, a read-only SELECT returning ONLY the entity primary-key column in a\n"
    "   ```sql fenced block. This is REQUIRED whenever the question restricts the entities to a named subset —\n"
    '   a tier, segment, region, status, or any attribute filter (e.g. "strategic customers", "active GPUs",\n'
    "   \"open orders\"): emit `SELECT <pk> FROM <entity_table> WHERE <attr> = '<value>'` so ONLY that subset is\n"
    "   scored and ranked. (Use the attribute values from the Columns section.) Without it the model scores a\n"
    "   default sample of ALL entities and the requested subset may not appear in the ranked results. Omit the\n"
    "   block only when the question is genuinely about all entities."
)


def build_pql_prompt(
    *,
    graph_ddl: str,
    columns: str = "",
    docs: list[str] | None = None,
    examples: list[dict[str, str]] | None = None,
    question: str,
    explain_entity: str | None = None,
    prev_pql: str | None = None,
    prev_error: str | None = None,
    dialect: str | None = None,
) -> str:
    """Assemble the text-to-PQL prompt (pure Python; mirrors the old Jinja template)."""
    docs = docs or []
    examples = examples or []
    sections: list[str] = [_PQL_HEADER]

    sections.append("## Graph (tables, keys, links)\n```\n" + graph_ddl + "\n```")

    # Snowflake stores unquoted identifiers uppercase, so the graph tables are
    # uppercase — require the LLM to match that case or the PQL won't parse.
    if (dialect or "").lower() == "snowflake":
        sections.append(_SNOWFLAKE_CASE_NOTE)

    if columns:
        sections.append("## Columns (types, values, notes)\n" + columns)

    if docs:
        sections.append("## Domain rules\n" + "\n".join(f"- {d}" for d in docs))

    if examples:
        blocks = []
        for ex in examples:
            block = f"Q: {ex['question']}\n```pql\n{ex['query']}\n```"
            reasoning = ex.get("reasoning")
            if reasoning:
                block += f"\nWhy: {reasoning}"
            blocks.append(block)
        sections.append("## Verified examples (question → PQL)\n" + "\n".join(blocks))

    if prev_pql:
        sections.append(
            "## Your previous attempt was invalid — fix it\n"
            "Previous PQL:\n```pql\n" + prev_pql + "\n```\n"
            "Error:\n" + (prev_error or "") + "\n\n" + _REPAIR_BODY
        )

    if explain_entity:
        safe = explain_entity.replace("'", "''")
        sections.append(
            "## Explanation entity\n"
            f"The tool call already supplies the exact entity id to explain: `{explain_entity}`. Generate the predictive\n"
            f"target and scope the PQL itself with `FOR <entity_table>.<primary_key>='{safe}'`.\n"
            "This is a single-entity query: NEVER emit `FOR EACH` and never copy an entity id from an example. The tool also\n"
            "passes this same id separately to KumoRFM after verifying its exact primary-key value in the warehouse."
        )

    output_lines = ["## Output format", "1. The PQL in a ```pql fenced block."]
    if not explain_entity:
        output_lines.append(_ENTITY_SQL_INSTRUCTION)
    sections.append("\n".join(output_lines))

    sections.append("## Question\n" + question)

    return "\n\n".join(sections)
