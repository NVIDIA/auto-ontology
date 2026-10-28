# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write the semantic description of every ColumnAttribute once compilation is done.

The taxonomy pass generates no descriptions: the catalog keeps only what a
source supplied, in ``description`` and ``value_description``. This pass runs
last, after FK resolution and the bridge-table stage, when every join a table
takes part in is known. One LLM call per table writes the description of each
non-FK column's ColumnAttribute, given:

* the column's catalog description and value description, which stay
  authoritative for meaning and value codes;
* its profiled samples, storage format and value shape;
* the table's key columns and the column each one joins to;
* the tables one join away, with their columns; and
* deterministic hints for columns whose name suggests a joined table's key while
  the join actually runs through a different column — ``orders."Store Code"``
  is not ``stores.StoreKey``.

FK columns are context only. They carry no ColumnAttribute, and their meaning
is the column they point at.

The catalog is never written, and nothing is re-embedded: an attribute's
embedding is built from the catalog description, not from this text. The
SQL generator shows this text for the columns semantic retrieval matched.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from auto_ontology.dal.descriptions import (
    fetch_neighbour_tables,
    fetch_table_for_enrichment,
    fetch_table_links,
    fetch_tables_with_term,
    write_attribute_descriptions,
)
from auto_ontology.semantic.cancellation import is_cancelled
from auto_ontology.semantic.models import ColumnDescriptionsResult
from auto_ontology.utils.llm_invoke import (
    get_llm_client,
    invoke_with_structured_output,
)
from auto_ontology.utils.sample_values import stringify_sample_values

logger = logging.getLogger(__name__)

_WORKERS = 3
# Wide tables overflow one structured-output response, so columns are described
# in batches that share the table's context.
_BATCH_SIZE = 15
_MAX_NEIGHBOURS = 6
_MAX_NEIGHBOUR_COLUMNS = 25
_MAX_SAMPLES = 5
_MAX_NEIGHBOUR_SAMPLES = 3
_MAX_SAMPLE_LEN = 40
_MAX_HINTS_PER_COLUMN = 3

# Words too common in column names to say two columns are about the same thing.
_GENERIC_TOKENS = frozenset(
    {
        "id",
        "code",
        "name",
        "type",
        "no",
        "num",
        "number",
        "key",
        "value",
        "date",
        "desc",
        "description",
        "status",
        "flag",
        "the",
        "of",
    }
)
# Words that make a column read as an identifier of something.
_KEY_TOKENS = frozenset({"id", "code", "key", "no", "num", "number"})

_SYSTEM_PROMPT = """\
You document a relational database for a semantic layer that a text-to-SQL \
agent reads. Write the description of each column listed under "Columns to \
describe".

Rules:
- An existing description and value description are authoritative for \
meaning. Keep every value code and mapping they state exactly (for example \
"1 = active, 0 = closed"), and drop only noise such as repeated punctuation or \
annotator notes.
- Use the samples, format and value shape as evidence of how values are \
written, and say it when it matters for filtering (for example "7-digit code \
stored as text", "Y/N flag", "date written YYYYMMDD").
- Use the key columns and joined tables to explain what a column refers to. \
When a hint says a column looks like a joined table's key but the join uses \
another column, say what the column is and that it is not that key.
- When the existing descriptions and the data disagree, describe what the \
data shows.
- Each description must read as a standalone fact about the data. Never refer \
to the existing descriptions, the samples, the hints or this prompt, and never \
say that something was or was not provided.
- One or two plain sentences per column. No SQL, no speculation beyond the \
evidence.
- Return exactly one entry per column to describe, using its physical name."""


def enrich_descriptions(database_name: str, table_ids: list[str]) -> int:
    """Describe the attributes of the tables in *table_ids* that have a Term.

    *table_ids* is the work list the taxonomy pass started from — tables that
    had no Term yet — so a table is enriched in the same compilation that first
    processes it and never again. Those still without a Term are skipped and
    come back on the next compilation together with the taxonomy pass.

    Returns the number of tables whose attributes were described.
    """
    with_term = fetch_tables_with_term(table_ids)
    work = [table_id for table_id in table_ids if table_id in with_term]
    if not work:
        logger.info("Description enrichment: no compiled tables to enrich")
        return 0

    logger.info("Description enrichment: %d table(s)", len(work))
    enriched = 0
    with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
        futures = {pool.submit(_enrich_one, table_id): table_id for table_id in work}
        for future in as_completed(futures):
            if future.result():
                enriched += 1
    logger.info(
        "Description enrichment finished (database=%r): %d of %d table(s) described",
        database_name,
        enriched,
        len(work),
    )
    return enriched


def _enrich_one(table_id: str) -> bool:
    if is_cancelled():
        return False
    try:
        return enrich_table_descriptions(table_id)
    except Exception:
        logger.exception("Description enrichment failed for table %s", table_id)
        return False


def enrich_table_descriptions(table_id: str) -> bool:
    """Describe one table's attributes; ``True`` when any was written."""
    table = fetch_table_for_enrichment(table_id)
    if table is None:
        return False
    targets = [c for c in table["columns"] if c.get("attribute_id")]
    if not targets:
        logger.info("[%s] no non-FK columns — skipping enrichment", table["name"])
        return False

    links = fetch_table_links(table_id)
    neighbour_ids = list(dict.fromkeys(str(link["other_table_id"]) for link in links))
    neighbours = fetch_neighbour_tables(neighbour_ids[:_MAX_NEIGHBOURS])
    context = build_table_context(table, links, neighbours)

    by_name: dict[str, str] = {}
    for start in range(0, len(targets), _BATCH_SIZE):
        batch = targets[start : start + _BATCH_SIZE]
        result = _describe(context, batch, table, links, neighbours)
        if result is not None:
            by_name.update(resolve_column_names(result, [c["name"] for c in batch]))

    descriptions = {
        str(column["attribute_id"]): by_name[column["name"]]
        for column in targets
        if by_name.get(column["name"])
    }
    if not descriptions:
        logger.warning("[%s] enrichment produced no descriptions", table["name"])
        return False

    written = write_attribute_descriptions(descriptions)
    logger.info(
        "[%s] described %d of %d attribute(s)",
        table["name"],
        len(written),
        len(targets),
    )
    return bool(written)


def _describe(
    context: str,
    batch: list[dict[str, Any]],
    table: dict[str, Any],
    links: list[dict[str, Any]],
    neighbours: dict[str, dict[str, Any]],
) -> ColumnDescriptionsResult | None:
    prompt = f"{context}\n\nColumns to describe:\n" + "\n".join(
        _target_line(c) for c in batch
    )
    hints = collision_hints(batch, table, links, neighbours)
    if hints:
        prompt += "\n\nHints:\n" + "\n".join(f"- {hint}" for hint in hints)
    try:
        return invoke_with_structured_output(
            get_llm_client(temperature=0.0),
            [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=prompt)],
            ColumnDescriptionsResult,
        )
    except Exception:
        logger.warning(
            "[%s] description batch errored — skipping it",
            table["name"],
            exc_info=True,
        )
        return None


def build_table_context(
    table: dict[str, Any],
    links: list[dict[str, Any]],
    neighbours: dict[str, dict[str, Any]],
) -> str:
    """The prompt section every batch of one table shares."""
    lines = [f"Table: {_qualified(table)}"]
    if table.get("description"):
        lines.append(f"Description: {table['description']}")

    by_name = {c["name"]: c for c in table["columns"]}
    key_lines: list[str] = []
    for link in links:
        local = by_name.get(link["local_column"]) or {}
        arrow = "->" if link["direction"] == "out" else "<-"
        line = (
            f"  - {link['local_column']} {arrow} "
            f"{link['other_table_name']}.{link['other_column']}"
        )
        shape = value_shape(_samples(local), local.get("data_type"))
        if shape:
            line += f" ({shape})"
        key_lines.append(line)
    if key_lines:
        lines.append(
            "Joins (-> this column points at the other table, <- the other "
            "table points here):"
        )
        lines.extend(dict.fromkeys(key_lines))

    for neighbour in neighbours.values():
        lines.append(f"\nJoined table: {neighbour['name']}")
        if neighbour.get("description"):
            lines.append(f"  Description: {neighbour['description']}")
        for column in neighbour["columns"][:_MAX_NEIGHBOUR_COLUMNS]:
            lines.append(_neighbour_line(column))
    return "\n".join(lines)


def _target_line(column: dict[str, Any]) -> str:
    parts = [f"  - {column['name']} ({column.get('data_type') or 'unknown'})"]
    if column.get("description"):
        parts.append(f"existing description: {column['description']}")
    if column.get("value_description"):
        parts.append(f"value description: {column['value_description']}")
    samples = _samples(column)
    if samples:
        parts.append(f"samples: {', '.join(samples)}")
    if column.get("format"):
        parts.append(f"format: {column['format']}")
    shape = value_shape(samples, column.get("data_type"))
    if shape:
        parts.append(f"shape: {shape}")
    if column.get("is_unique"):
        parts.append("unique in the sample")
    return " | ".join(parts)


def _neighbour_line(column: dict[str, Any]) -> str:
    line = f"  - {column['name']} ({column.get('data_type') or 'unknown'})"
    if column.get("description"):
        line += f": {column['description']}"
    samples = _samples(column)[:_MAX_NEIGHBOUR_SAMPLES]
    if samples:
        line += f" | samples: {', '.join(samples)}"
    return line


def collision_hints(
    columns: list[dict[str, Any]],
    table: dict[str, Any],
    links: list[dict[str, Any]],
    neighbours: dict[str, dict[str, Any]],
) -> list[str]:
    """Columns that look like a joined table's key but are not the join.

    A column is flagged when its name is only a joined table's name plus a key
    word (``Store Code`` next to a join to ``stores``), or both columns read as
    identifiers sharing a distinctive word while written differently, and the
    join between the two tables runs through other columns. The hint states
    both value shapes, which is usually what tells the two apart.
    """
    joined_columns: dict[str, set[str]] = {}
    join_pairs: dict[str, list[tuple[str, str]]] = {}
    for link in links:
        other = str(link["other_table_id"])
        joined_columns.setdefault(other, set()).add(link["local_column"])
        join_pairs.setdefault(other, []).append(
            (link["local_column"], link["other_column"])
        )

    hints: list[str] = []
    for column in columns:
        name = column["name"]
        all_tokens = name_tokens(name)
        tokens = all_tokens - _GENERIC_TOKENS
        if not tokens:
            continue
        is_key_like = bool(all_tokens & _KEY_TOKENS)
        own_shape = value_shape(_samples(column), column.get("data_type"))
        found: list[str] = []
        for table_id, neighbour in neighbours.items():
            if name in joined_columns.get(table_id, set()):
                continue
            pairs = join_pairs.get(table_id) or []
            join_text = ", ".join(
                f"{table['name']}.{local} = {neighbour['name']}.{other}"
                for local, other in pairs
            )
            if is_key_like and tokens <= _table_tokens(neighbour["name"]):
                found.append(
                    f"{name} names table {neighbour['name']}, but the join is "
                    f"{join_text}" + _shapes(own_shape, neighbour, pairs)
                )
            for other in neighbour["columns"]:
                if other["name"] in {p[1] for p in pairs}:
                    continue
                other_tokens = name_tokens(other["name"])
                if (
                    is_key_like
                    and other_tokens & _KEY_TOKENS
                    and tokens & (other_tokens - _GENERIC_TOKENS)
                ):
                    other_shape = value_shape(_samples(other), other.get("data_type"))
                    if own_shape and other_shape and own_shape != other_shape:
                        found.append(
                            f"{name} ({own_shape}) resembles "
                            f"{neighbour['name']}.{other['name']} ({other_shape}); "
                            f"the tables join on {join_text}"
                        )
        hints.extend(found[:_MAX_HINTS_PER_COLUMN])
    return hints


def _shapes(
    own_shape: str | None,
    neighbour: dict[str, Any],
    pairs: list[tuple[str, str]],
) -> str:
    by_name = {c["name"]: c for c in neighbour["columns"]}
    shapes = []
    for _local, other in pairs:
        key = by_name.get(other)
        shape = key and value_shape(_samples(key), key.get("data_type"))
        if shape:
            shapes.append(f"{neighbour['name']}.{other} is {shape}")
    if not own_shape and not shapes:
        return ""
    own = f"this column is {own_shape}" if own_shape else ""
    return " (" + "; ".join(p for p in [own, *shapes] if p) + ")"


def name_tokens(name: str) -> set[str]:
    """Lower-case words of a column or table name: camelCase, snake and spaces."""
    spaced = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", name)
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", spaced)
    return {t.lower() for t in re.split(r"[^A-Za-z0-9]+", spaced) if t}


def _table_tokens(name: str) -> set[str]:
    tokens = name_tokens(name)
    singular = {t[:-1] for t in tokens if len(t) > 3 and t.endswith("s")}
    return (tokens | singular) - _GENERIC_TOKENS


def value_shape(samples: list[str], data_type: str | None) -> str | None:
    """How the sampled values are written, when they share one shape."""
    values = [v for v in samples if v != ""]
    if not values:
        return None
    is_text = "char" in (data_type or "").lower() or "text" in (data_type or "").lower()
    lengths = sorted({len(v) for v in values})
    span = f"{lengths[0]}" if len(lengths) == 1 else f"{lengths[0]}-{lengths[-1]}"
    if all(v.isdigit() for v in values):
        suffix = " stored as text" if is_text else ""
        return f"{span} digits{suffix}"
    if is_text and all(re.fullmatch(r"-?\d+(\.\d+)?", v) for v in values):
        return "numbers stored as text"
    if all(v.isalpha() and v.isupper() for v in values) and lengths[-1] <= 3:
        return f"{span}-letter upper-case codes"
    if len(lengths) == 1 and lengths[0] <= 20:
        return f"always {lengths[0]} characters"
    return None


def resolve_column_names(
    result: ColumnDescriptionsResult, requested: list[str]
) -> dict[str, str]:
    """Map the model's column names back to the physical names requested.

    The prompt annotates each column as ``<name> (<type>)``, and a model given a
    name with spaces or parentheses sometimes echoes the annotation back
    (``"Order Date (TEXT)"``). Exact match first, then the longest requested
    name the returned string starts with.
    """
    requested_set = set(requested)
    out: dict[str, str] = {}
    for entry in result.columns:
        raw = entry.column_name or ""
        text = (entry.description or "").strip()
        if not text:
            continue
        if raw in requested_set:
            out.setdefault(raw, text)
            continue
        prefixes = [n for n in requested if n and raw.startswith(n)]
        if prefixes:
            out.setdefault(max(prefixes, key=len), text)
    return out


def _samples(column: dict[str, Any]) -> list[str]:
    values = stringify_sample_values(
        column.get("sample_values"), max_len=_MAX_SAMPLE_LEN
    )
    return (values or [])[:_MAX_SAMPLES]


def _qualified(table: dict[str, Any]) -> str:
    schema = table.get("schema_name")
    return f"{schema}.{table['name']}" if schema else table["name"]
