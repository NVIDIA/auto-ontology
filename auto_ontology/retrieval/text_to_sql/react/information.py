# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``retrieve(purpose="information")``: what the data holds, as itself.

A query needs tables and columns, so the SQL view of ``retrieve`` shows every
term and attribute it finds as the table that stores it. A question about the
data itself — what a term means, which tables there are — needs the opposite:
the term with its description, synonyms, attributes and definitions, or the
whole list of tables or terms, which a search capped at a few tables cannot
give. Same searches, same database scope, a different rendering.
"""

from __future__ import annotations

from typing import Any

from auto_ontology.catalog.constants import Labels
from auto_ontology.dal.attributes import fetch_attr_column_contexts
from auto_ontology.dal.datasources import (
    fetch_databases,
    fetch_schema_ids_for_database,
    fetch_schemas_for_database,
    fetch_tables_by_ids,
    fetch_tables_for_schema,
)
from auto_ontology.dal.sql_attributes import fetch_sql_attributes_by_term_id
from auto_ontology.dal.terms import (
    fetch_all_terms,
    fetch_column_attributes_by_term_id,
    fetch_term_table_pairs,
)
from auto_ontology.retrieval.text_to_sql.react import config
from auto_ontology.retrieval.text_to_sql.react.context import ToolContext
from auto_ontology.retrieval.text_to_sql.react.retrieve import (
    _definition_block,
    _meaning_hits,
    _merge_hits,
    _name_hits,
    _uncovered_physical_hits,
    resolve_kinds,
)
from auto_ontology.semantic.constants import LABEL_COLUMN_ATTRIBUTE, LABEL_TERM

PURPOSES = ("sql", "information")

_DESCRIPTION_CHARS = 200
_SQL_CHARS = 600
#: Terms shown with their attributes and definitions; the rest get one line.
_FULL_TERMS = 2


def _clip(text: Any, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _cut(block: str, budget: int) -> str:
    """*block* shortened to *budget* characters, at a line end when it can be."""
    kept = block[: max(budget - 1, 0)]
    if "\n" in kept:
        kept = kept[: kept.rindex("\n")]
    return kept.rstrip() + "…"


def _fit(blocks: list[str], noun: str, hint: str, sep: str) -> str:
    """*blocks* up to the observation limit, with a count of the rest.

    The first block is always shown, cut to the room there is when it alone is
    too long. Narrowing the query cannot shrink a block that matches again, so
    dropping it would leave the model with nothing it could act on.
    """
    limit = config.max_observation_chars()
    budget = max(limit - 120, 1)
    shown: list[str] = []
    used = 0
    for block in blocks:
        if used + len(block) + len(sep) > budget:
            break
        shown.append(block)
        used += len(block) + len(sep)
    if not shown and blocks:
        shown.append(_cut(blocks[0], budget))
    rest = len(blocks) - len(shown)
    if rest:
        shown.append(f"… {rest} more {noun} not shown; {hint}.")
    return sep.join(shown)


def _schema_ids(ctx: ToolContext) -> list[str]:
    if ctx.database_name:
        return fetch_schema_ids_for_database(ctx.database_name)
    return [
        str(schema["id"])
        for database in fetch_databases()
        for schema in (fetch_schemas_for_database(str(database["id"])) or {}).get(
            "schemas", []
        )
    ]


def _database_terms(ctx: ToolContext) -> dict[str, dict[str, Any]]:
    terms = {str(t["id"]): t for t in fetch_all_terms()}
    if not ctx.database_name:
        return terms
    in_database = {
        str(pair["term_id"])
        for pair in fetch_term_table_pairs(database_name=ctx.database_name)
    }
    return {tid: t for tid, t in terms.items() if tid in in_database}


def _term_line(term: dict[str, Any]) -> str:
    line = f"TERM {term.get('name') or ''}"
    if term.get("description"):
        line += f" — {_clip(term['description'], _DESCRIPTION_CHARS)}"
    if synonyms := [s for s in term.get("synonyms") or [] if s]:
        line += f" (also: {', '.join(synonyms)})"
    return line


def _term_block(ctx: ToolContext, term: dict[str, Any]) -> str:
    """One term with the tables that store it, its attributes and definitions."""
    term_id = str(term["id"])
    table_ids = [
        str(pair["table_id"])
        for pair in fetch_term_table_pairs(
            database_name=ctx.database_name, term_ids=[term_id]
        )
    ]
    tables = {
        str(t["id"]): f"{t['schema_name']}.{t['name']}"
        for t in fetch_tables_by_ids(list(dict.fromkeys(table_ids)))
    }
    lines = [_term_line(term)]
    if tables:
        lines.append("  Stored in: " + ", ".join(sorted(tables.values())))
    attributes = [
        a
        for a in fetch_column_attributes_by_term_id(term_id)
        if not tables or str(a.get("table_id")) in tables
    ]
    if attributes:
        lines.append("  Attributes:")
    for a in attributes:
        column = a.get("primary_column") or {}
        where = tables.get(str(a.get("table_id")), column.get("table_name") or "")
        text = f"    {a.get('name')} ({where}.{column.get('column_name') or a.get('source_column')})"
        if a.get("description"):
            text += f" — {_clip(a['description'], _DESCRIPTION_CHARS)}"
        lines.append(text)
    definitions = fetch_sql_attributes_by_term_id(term_id)
    if definitions:
        lines.append("  Definitions:")
    for d in definitions:
        text = f"    {d.get('name')}"
        if d.get("description"):
            text += f" — {_clip(d['description'], _DESCRIPTION_CHARS)}"
        lines.append(text)
        if d.get("sql"):
            lines.append(f"      SQL: {_clip(d['sql'], _SQL_CHARS)}")
    return "\n".join(lines)


def _attribute_line(context: dict[str, Any]) -> str:
    where = ".".join(
        p
        for p in (
            context.get("schema_name"),
            context.get("table_name"),
            context.get("col_name"),
        )
        if p
    )
    line = f"ATTRIBUTE {context.get('attr_name')}"
    if context.get("term_name"):
        line += f" (of {context['term_name']})"
    line += f" — column {where}"
    if context.get("attr_description"):
        line += f": {_clip(context['attr_description'], _DESCRIPTION_CHARS)}"
    return line


def _column_line(ctx: ToolContext, hit: dict[str, Any]) -> str | None:
    """A column no attribute stands for, with its table; ``None`` off-database."""
    table = next(
        (t for t in hit.get("relevant_tables") or [] if isinstance(t, dict)), None
    )
    if table is None or (
        ctx.database_name and table.get("database_name") != ctx.database_name
    ):
        return None
    name = hit.get("name")
    column = next((c for c in table.get("columns") or [] if c.get("name") == name), hit)
    where = ".".join(
        str(p) for p in (table.get("schema_name"), table.get("name"), name) if p
    )
    line = f"COLUMN {where}"
    if column.get("data_type"):
        line += f" ({column['data_type']})"
    details = []
    if column.get("description"):
        details.append(_clip(column["description"], _DESCRIPTION_CHARS))
    if column.get("sample_values"):
        details.append(f"values: {_clip(column['sample_values'], _DESCRIPTION_CHARS)}")
    return line + (f" — {'; '.join(details)}" if details else "")


def _table_line(table: dict[str, Any]) -> str:
    line = f"TABLE {table.get('schema_name')}.{table.get('name')}"
    if table.get("description"):
        line += f" — {_clip(table['description'], _DESCRIPTION_CHARS)}"
    return line


def _list_all(ctx: ToolContext, kinds: list[str] | None) -> str:
    if "term" in [k.lower() for k in kinds or []]:
        lines = sorted(
            (_term_line(t) for t in _database_terms(ctx).values()), key=str.casefold
        )
        noun = "terms"
    else:
        lines = sorted(
            (
                _table_line(t)
                for schema_id in _schema_ids(ctx)
                for t in fetch_tables_for_schema(schema_id)
            ),
            key=str.casefold,
        )
        noun = "tables"
    if not lines:
        return f"No {noun} found."
    return _fit(lines, noun, "search with words to narrow them", "\n")


def information(ctx: ToolContext, query: str, kinds: list[str] | None = None) -> str:
    """What *query* finds, shown as itself; with no query, every table or term."""
    query = (query or "").strip()
    if not query:
        return _list_all(ctx, kinds)
    labels = resolve_kinds(kinds)
    hits = _merge_hits(
        _meaning_hits(ctx, query, labels),
        _uncovered_physical_hits(ctx, query, labels),
        _name_hits(ctx, query, labels),
    )
    terms = (
        _database_terms(ctx) if any(h.get("label") == LABEL_TERM for h in hits) else {}
    )
    attr_ids = [str(h["id"]) for h in hits if h.get("label") == LABEL_COLUMN_ATTRIBUTE]
    # The lookup keeps an attribute whose column is in another database, with an
    # empty ``database_name``, so the database has to be checked here.
    attributes = {
        attr_id: context
        for attr_id, context in (
            fetch_attr_column_contexts(attr_ids, database_name=ctx.database_name)
            if attr_ids
            else {}
        ).items()
        if not ctx.database_name or context.get("database_name") == ctx.database_name
    }
    table_ids = [str(h["id"]) for h in hits if h.get("label") == Labels.TABLE]
    tables = {
        str(t["id"]): t
        for t in (fetch_tables_by_ids(table_ids) if table_ids else [])
        if not ctx.database_name or t.get("database_name") == ctx.database_name
    }

    blocks: list[str] = []
    full_terms: set[str] = set()
    for hit in hits:
        hit_id = str(hit["id"])
        label = hit.get("label")
        if label == LABEL_TERM and hit_id in terms:
            if len(full_terms) < _FULL_TERMS:
                full_terms.add(str(terms[hit_id].get("name")))
                blocks.append(_term_block(ctx, terms[hit_id]))
            else:
                blocks.append(_term_line(terms[hit_id]))
        elif label == LABEL_COLUMN_ATTRIBUTE and hit_id in attributes:
            if attributes[hit_id].get("term_name") not in full_terms:
                blocks.append(_attribute_line(attributes[hit_id]))
        elif label == Labels.TABLE and hit_id in tables:
            blocks.append(_table_line(tables[hit_id]))
        elif label == Labels.COLUMN:
            if (line := _column_line(ctx, hit)) is not None:
                blocks.append(line)
        elif (block := _definition_block(hit)) is not None:
            blocks.append(block)
    blocks = list(dict.fromkeys(blocks))
    if not blocks:
        return (
            f"No matches for {query!r}. Try other words, or an empty query to "
            'list every table (kinds=["term"] for every term).'
        )
    return _fit(blocks, "matches", "narrow the query to see them", "\n\n")
