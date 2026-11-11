# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ReAct agent's ``retrieve`` tool: free-text search over the ontology.

The fixed pipeline searches once, on entities an LLM extracted from the
question, and then filters the hits with more LLM calls. This tool takes the
opposite position: the model writes the query — the question, a follow-up such
as "how is customer ID calculated", or a bare column name — and the tool runs
the same searches the pipeline runs, with no LLM inside. What to look for, and
whether what came back is relevant, are the policy's decisions, so they are the
ones training can improve.

One call does three things, all through existing data-access functions:

1. **Search by meaning** — :func:`_get_candidates_information` over the
   semantic layer: Terms stand for tables and ColumnAttributes for columns,
   since their text is the processed, natural-language description of the
   data; definitions and analyses are searched as themselves. Tables and
   columns the semantic layer has no row for — foreign keys, all-key link
   tables — are searched in the data collection instead.
2. **Search by name** — :func:`fetch_global_search`, the catalog's substring
   search, which finds codes an embedding misses (``CDSCode``, ``T-BIL``). A
   word that looks copied from a schema is also matched against physical table
   and column names, which attribute display names do not carry.
3. **Expand one step along the graph** — a term's tables
   (:func:`fetch_term_table_pairs`) and an attribute's owning column
   (:func:`fetch_attr_column_contexts`), each table with its columns, stored
   values and semantic FKs (:func:`fetch_tables_by_ids`), the join route from
   every new table to the tables earlier calls already showed
   (:func:`find_join_path`), and the near-copies of every new table
   (:mod:`.siblings`).

The model sees only tables and columns, with their descriptions.

The result is rendered as text and capped in size. The same query against the
same catalog returns the same text.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import sqlglot
from sqlglot import exp

from auto_ontology.catalog.constants import Labels
from auto_ontology.dal.attributes import fetch_attr_column_contexts, find_join_path
from auto_ontology.dal.candidates import expand_info
from auto_ontology.dal.datasources import (
    fetch_schema_ids_for_database,
    fetch_tables_by_ids,
    find_column_id_by_table_and_name,
)
from auto_ontology.dal.search import fetch_global_search, search_tokens
from auto_ontology.dal.terms import fetch_semantic_coverage, fetch_term_table_pairs
from auto_ontology.retrieval.data_access.candidates import _get_candidates_information
from auto_ontology.retrieval.text_to_sql.react import config
from auto_ontology.retrieval.text_to_sql.react.context import ToolContext
from auto_ontology.retrieval.text_to_sql.react.siblings import sibling_lines
from auto_ontology.semantic.constants import (
    LABEL_COLUMN_ATTRIBUTE,
    LABEL_SQL_ATTRIBUTE,
    LABEL_TERM,
    REL_REPRESENTS,
)

logger = logging.getLogger(__name__)

#: What the model may narrow a search to, and the semantic labels each covers.
KIND_LABELS: dict[str, list[str]] = {
    "term": [LABEL_TERM],
    "attribute": [LABEL_COLUMN_ATTRIBUTE],
    "definition": [LABEL_SQL_ATTRIBUTE],
    "analysis": [Labels.CUSTOM_ANALYSIS],
}

#: The physical names an identifier-like word is also matched against.
_PHYSICAL_LABELS: dict[str, str] = {
    LABEL_TERM: Labels.TABLE,
    LABEL_COLUMN_ATTRIBUTE: Labels.COLUMN,
}

#: Hits kept per label from the meaning search. Attributes get the most because
#: columns are what a query is written from; definitions are few and long.
_PER_LABEL_K: dict[str, int] = {
    LABEL_TERM: 8,
    LABEL_COLUMN_ATTRIBUTE: 15,
    LABEL_SQL_ATTRIBUTE: 3,
    Labels.CUSTOM_ANALYSIS: 3,
}

#: Physical rows searched, then kept, for what the semantic layer leaves out.
#: More are searched than kept because covered rows are dropped after ranking,
#: and would otherwise crowd an uncovered one out of a small top k.
_PHYSICAL_SEARCH_K: dict[str, int] = {Labels.TABLE: 10, Labels.COLUMN: 20}
_PHYSICAL_KEEP: dict[str, int] = {Labels.TABLE: 3, Labels.COLUMN: 5}

_NAME_SEARCH_LIMIT = 10
_TERM_TABLE_LIMIT = 3
_JOIN_PAIRS_PER_CALL = 10
_MAX_DESCRIPTION_CHARS = 160
_MAX_DEFINITION_CHARS = 500
_MAX_SAMPLE_VALUES = 5

# A word the model most likely copied from a schema rather than wrote in
# English: it has an underscore, a hyphen, a digit, or a capital inside it.
_IDENTIFIER_RE = re.compile(r"^(?=.*[_\-\d]|.+[A-Z]).+$")


def resolve_kinds(kinds: list[str] | None) -> list[str]:
    """Labels to search for *kinds*; every label when none or none are known."""
    labels = [
        label for kind in (kinds or []) for label in KIND_LABELS.get(kind.lower(), [])
    ]
    return labels or [label for labels in KIND_LABELS.values() for label in labels]


def _identifier_words(query: str) -> list[str]:
    words = [w.strip("\"'`.,;:()[]?") for w in query.split()]
    return list(dict.fromkeys(w for w in words if w and _IDENTIFIER_RE.match(w)))


def _meaning_hits(ctx: ToolContext, query: str, labels: list[str]) -> list[dict]:
    return list(
        _get_candidates_information(
            ctx.retriever,
            query,
            list_of_semantic=labels,
            database_name=ctx.database_name,
            per_label_k={label: _PER_LABEL_K.get(label, 5) for label in labels},
        )
    )


def _uncovered_physical_hits(
    ctx: ToolContext, query: str, labels: list[str]
) -> list[dict]:
    """Tables and columns by meaning that no Term or ColumnAttribute stands for.

    Foreign-key columns get no attribute and all-key link tables no Term, so
    the semantic search alone could never land on them.
    """
    physical = [_PHYSICAL_LABELS[lb] for lb in labels if lb in _PHYSICAL_LABELS]
    if not physical or ctx.data_retriever is None:
        return []
    hits = list(
        _get_candidates_information(
            ctx.data_retriever,
            query,
            list_of_semantic=physical,
            database_name=ctx.database_name,
            per_label_k={label: _PHYSICAL_SEARCH_K[label] for label in physical},
        )
    )
    covered_columns, covered_tables = fetch_semantic_coverage(
        [str(h["id"]) for h in hits if h.get("label") == Labels.COLUMN],
        [str(h["id"]) for h in hits if h.get("label") == Labels.TABLE],
    )
    covered = {Labels.COLUMN: covered_columns, Labels.TABLE: covered_tables}
    kept: list[dict] = []
    counts = dict.fromkeys(physical, 0)
    for hit in hits:
        label = hit.get("label")
        if label not in counts or str(hit.get("id")) in covered[label]:
            continue
        if counts[label] < _PHYSICAL_KEEP[label]:
            counts[label] += 1
            kept.append(hit)
    return kept


def _name_scopes(
    object_types: set[str], schema_ids: list[str]
) -> list[tuple[set[str], list[str] | None]]:
    """*object_types* split into the physical ones, under *schema_ids*, and the rest."""
    if not schema_ids:
        return [(object_types, None)]
    physical = object_types & set(_PHYSICAL_LABELS.values())
    rest = object_types - physical
    return [
        (types, ids) for types, ids in ((physical, schema_ids), (rest, None)) if types
    ]


def _name_hits(ctx: ToolContext, query: str, labels: list[str]) -> list[dict]:
    """Substring hits on names: the whole query, then each identifier in it.

    The whole query has to match every token in one name or one description,
    which a sentence rarely does — so a word that looks copied from a schema is
    also searched alone, by name only, and against physical names too.

    Physical names are searched in the target database only: the search keeps
    its first rows across the whole catalog, and with many databases sharing a
    table or column name the target's copy can fall outside them. Terms and
    attributes do not live under a schema, which the scope would rule out
    entirely; their tables are narrowed to the database when they are resolved.
    """
    semantic = set(labels)
    physical = semantic | {
        _PHYSICAL_LABELS[lb] for lb in labels if lb in _PHYSICAL_LABELS
    }
    searches = [(search_tokens(query), True, semantic)]
    searches += [
        (search_tokens(word), False, physical) for word in _identifier_words(query)
    ]
    schema_ids = (
        fetch_schema_ids_for_database(ctx.database_name) if ctx.database_name else []
    )

    raw: list[dict] = []
    for tokens, include_description, object_types in searches:
        if not tokens:
            continue
        for types, data_ids in _name_scopes(object_types, schema_ids):
            raw += fetch_global_search(
                tokens,
                types,
                include_description=include_description,
                data_ids=data_ids,
                limit=_NAME_SEARCH_LIMIT,
            )
    hits = [{"id": h["id"], "label": h["label"], "name": h.get("name")} for h in raw]
    extra = expand_info([{"id": h["id"], "label": h["label"]} for h in hits])
    for hit in hits:
        hit.update(extra.get(hit["id"]) or extra.get(str(hit["id"])) or {})
    return hits


def _merge_hits(*searches: list[dict]) -> list[dict]:
    """Hits in search order, each once."""
    merged: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for hit in [h for hits in searches for h in hits]:
        if hit.get("id") is None:
            continue
        key = (str(hit.get("label")), str(hit["id"]))
        if key in seen:
            continue
        seen.add(key)
        merged.append(hit)
    return merged


def _hit_table_ids(hit: dict) -> list[str]:
    return [
        str(table["id"])
        for table in hit.get("relevant_tables") or []
        if isinstance(table, dict) and table.get("id") is not None
    ]


def _resolve_tables(
    ctx: ToolContext, hits: list[dict]
) -> tuple[list[str], dict[str, set[str]]]:
    """Tables the hits live on, in hit order, plus which columns each matched.

    A term stands for the tables it represents, not the ones that merely
    reference it, and an attribute for the column that owns it.
    """
    attr_ids = [str(h["id"]) for h in hits if h.get("label") == LABEL_COLUMN_ATTRIBUTE]
    attr_contexts = (
        fetch_attr_column_contexts(attr_ids, database_name=ctx.database_name)
        if attr_ids
        else {}
    )
    term_ids = [str(h["id"]) for h in hits if h.get("label") == LABEL_TERM]
    term_tables: dict[str, list[str]] = {}
    if term_ids:
        for pair in fetch_term_table_pairs(
            term_ids=term_ids, database_name=ctx.database_name
        ):
            if pair.get("path") != REL_REPRESENTS:
                continue
            tables = term_tables.setdefault(str(pair["term_id"]), [])
            if len(tables) < _TERM_TABLE_LIMIT and str(pair["table_id"]) not in tables:
                tables.append(str(pair["table_id"]))

    table_ids: list[str] = []
    matched: dict[str, set[str]] = {}
    for hit in hits:
        label = hit.get("label")
        hit_id = str(hit["id"])
        if label == Labels.TABLE:
            ids = [hit_id]
        elif label == LABEL_COLUMN_ATTRIBUTE:
            context = attr_contexts.get(hit_id) or {}
            ids = [str(context["table_id"])] if context.get("table_id") else []
            if ids and context.get("col_name"):
                matched.setdefault(ids[0], set()).add(context["col_name"])
        elif label == LABEL_TERM:
            ids = term_tables.get(hit_id, [])
        else:
            ids = _hit_table_ids(hit)
            if label == Labels.COLUMN and ids and hit.get("name"):
                matched.setdefault(ids[0], set()).add(str(hit["name"]))
        table_ids += [tid for tid in ids if tid not in table_ids]
    return table_ids, matched


def _in_database(ctx: ToolContext, table: dict) -> bool:
    return not ctx.database_name or table.get("database_name") == ctx.database_name


def _qualified(table: dict) -> str:
    return ".".join(p for p in (table.get("schema_name"), table.get("name")) if p)


def _clip(text: Any, limit: int) -> str:
    collapsed = " ".join(str(text or "").split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 1] + "…"


def _sample_values(values: Any) -> str:
    if not values:
        return ""
    if isinstance(values, str):
        values = [values]
    shown = [repr(v) for v in list(values)[:_MAX_SAMPLE_VALUES]]
    return ", ".join(shown)


def _column_line(column: dict) -> str:
    parts = [f"  {column.get('name')} {column.get('data_type') or ''}".rstrip()]
    details = []
    if column.get("description"):
        details.append(_clip(column["description"], _MAX_DESCRIPTION_CHARS))
    if samples := _sample_values(column.get("sample_values")):
        details.append(f"values: {samples}")
    if column.get("format"):
        details.append(f"format: {column['format']}")
    if column.get("semantic_fk"):
        details.append(f"references {column['semantic_fk']}")
    if details:
        parts.append(" — " + "; ".join(details))
    return "".join(parts)


def _shown_columns(table: dict, matched: set[str]) -> tuple[list[dict], int]:
    """Columns to render for *table*, and how many were left out.

    A table at or under the column cap shows every column. A wider one shows
    the columns the search matched, the key, and the columns that reference
    another table — the ones a query is most likely to need — up to the cap.
    """
    columns = list(table.get("columns") or [])
    cap = config.max_columns_per_table()
    if len(columns) <= cap:
        return columns, 0
    keys = {str(k) for k in table.get("pk") or []}
    lowered = {m.lower() for m in matched}

    def priority(column: dict) -> int:
        name = str(column.get("name") or "")
        if name.lower() in lowered:
            return 0
        if name in keys:
            return 1
        return 2 if column.get("semantic_fk") else 3

    chosen = sorted((c for c in columns if priority(c) < 3), key=lambda c: priority(c))[
        :cap
    ]
    return chosen, len(columns) - len(chosen)


def _table_block(table: dict, matched: set[str], seen_before: bool) -> str:
    header = f"TABLE {_qualified(table)}"
    if table.get("description"):
        header += f" — {_clip(table['description'], _MAX_DESCRIPTION_CHARS)}"
    if seen_before:
        lowered = {m.lower() for m in matched}
        columns = [
            c
            for c in table.get("columns") or []
            if str(c.get("name")).lower() in lowered
        ]
        lines = [header + " (shown earlier; matched columns only)"]
        return "\n".join(lines + [_column_line(c) for c in columns])
    columns, hidden = _shown_columns(table, matched)
    lines = [header] + [_column_line(c) for c in columns]
    if hidden:
        lines.append(
            f"  … {hidden} more columns not shown; retrieve with a narrower query "
            "to see them"
        )
    return "\n".join(lines)


def _definition_block(hit: dict) -> str | None:
    """A line for a hit that carries meaning beyond a table: a definition."""
    label = hit.get("label")
    name = hit.get("name") or hit.get("text") or ""
    description = _clip(hit.get("description"), _MAX_DESCRIPTION_CHARS)
    if label == LABEL_SQL_ATTRIBUTE:
        line = f"ATTRIBUTE {name}"
        if hit.get("term_name"):
            line += f" (of {hit['term_name']})"
        if description:
            line += f" — {description}"
        if hit.get("sql"):
            line += f"\n  SQL: {_clip(hit['sql'], _MAX_DEFINITION_CHARS)}"
        return line
    if label == Labels.CUSTOM_ANALYSIS:
        line = f"ANALYSIS {name}" + (f" — {description}" if description else "")
        if hit.get("sql"):
            line += f"\n  SQL: {_clip(hit['sql'], _MAX_DEFINITION_CHARS)}"
        return line
    return None


def _anchor_column_id(ctx: ToolContext, table: dict) -> str | None:
    """Any column id on *table*, which is all :func:`find_join_path` needs."""
    table_id = str(table["id"])
    if table_id not in ctx.anchor_column_ids:
        columns = table.get("columns") or []
        ctx.anchor_column_ids[table_id] = (
            find_column_id_by_table_and_name(
                str(table.get("name")),
                str(columns[0].get("name")),
                table.get("database_name") or ctx.database_name,
            )
            if columns
            else None
        )
    return ctx.anchor_column_ids[table_id]


def _hop_text(hop: dict) -> str:
    left = ".".join(p for p in (hop.get("source_schema"), hop.get("source_table")) if p)
    right = ".".join(
        p for p in (hop.get("target_schema"), hop.get("target_table")) if p
    )
    return f"{left}.{hop.get('source_column')} = {right}.{hop.get('target_column')}"


def _hop_tables(hops: list[dict]) -> set[tuple[str | None, str | None]]:
    return {
        (hop.get(f"{side}_schema"), hop.get(f"{side}_table"))
        for hop in hops
        for side in ("source", "target")
    }


def _sql_tables(sql: str | None) -> set[tuple[str | None, str | None]]:
    """Tables a definition's SQL names; none when it does not parse."""
    if not sql:
        return set()
    try:
        statements = sqlglot.parse(sql)
    except sqlglot.errors.SqlglotError:
        return set()
    return {
        (table.db or None, table.name)
        for statement in statements
        if statement is not None
        for table in statement.find_all(exp.Table)
    }


def _join_lines(
    ctx: ToolContext, new_tables: list[dict]
) -> list[tuple[str, set[tuple[str | None, str | None]]]]:
    """Join routes from each new table to every table already in view.

    Each route comes with every table on it, bridges included.
    """
    lines: list[tuple[str, set[tuple[str | None, str | None]]]] = []
    lookups = 0
    others = list(ctx.seen_tables.values()) + new_tables
    for table in new_tables:
        for other in others:
            pair = frozenset((str(table["id"]), str(other["id"])))
            if len(pair) < 2 or pair in ctx.joined_pairs:
                continue
            if lookups >= _JOIN_PAIRS_PER_CALL:
                return lines
            lookups += 1
            ctx.joined_pairs.add(pair)
            anchor = _anchor_column_id(ctx, table)
            dest = _anchor_column_id(ctx, other)
            if not anchor or not dest:
                continue
            hops = find_join_path(anchor, dest)
            if hops:
                ctx.join_paths.append({"path": hops})
                line = f"JOIN {_qualified(table)} ↔ {_qualified(other)}: " + " → ".join(
                    _hop_text(h) for h in hops
                )
                lines.append((line, _hop_tables(hops)))
    return lines


def _cap(blocks: list[str]) -> str:
    """Join *blocks* in order within the observation limit, dropping repeats.

    A block that does not fit is skipped rather than ending the result, so the
    short lines after a large table still use the room that is left.
    """
    limit = config.max_observation_chars()
    out: list[str] = []
    seen: set[str] = set()
    used = 0
    skipped = 0
    for block in blocks:
        if block in seen:
            continue
        seen.add(block)
        if used + len(block) + 2 > limit:
            skipped += 1
            continue
        out.append(block)
        used += len(block) + 2
    if skipped:
        out.append(
            f"… {skipped} more results not shown; narrow the query or pass kinds "
            "to see them"
        )
    return "\n\n".join(out)


def retrieve(ctx: ToolContext, query: str, kinds: list[str] | None = None) -> str:
    """Search the ontology for *query* and render what was found."""
    query = (query or "").strip()
    if not query:
        return "Error: query is empty. Say what you are looking for."
    labels = resolve_kinds(kinds)
    hits = _merge_hits(
        _meaning_hits(ctx, query, labels),
        _uncovered_physical_hits(ctx, query, labels),
        _name_hits(ctx, query, labels),
    )

    table_ids, matched = _resolve_tables(ctx, hits)
    new_ids = [tid for tid in table_ids if tid not in ctx.seen_tables]
    # Filtered before the limit, so a table from another database never takes
    # the place of one from this database.
    fetched = {str(t["id"]): t for t in fetch_tables_by_ids(new_ids)}
    new_tables = [
        fetched[tid]
        for tid in new_ids
        if tid in fetched and _in_database(ctx, fetched[tid])
    ][: config.max_tables_per_retrieve()]
    revisited = [
        ctx.seen_tables[tid]
        for tid in table_ids
        if tid in ctx.seen_tables and matched.get(tid)
    ]

    def own(table: dict) -> set[tuple[str | None, str | None]]:
        return {(table.get("schema_name"), table.get("name"))}

    # Most useful first, since the limit cuts from the end: new tables, how they
    # join, definitions that carry SQL, then tables seen before. Each block
    # carries the tables it names.
    new_blocks = {
        str(t["id"]): _table_block(t, matched.get(str(t["id"]), set()), False)
        for t in new_tables
    }
    named: list[tuple[str, set[tuple[str | None, str | None]]]] = [
        (new_blocks[str(t["id"])], own(t)) for t in new_tables
    ]
    named += _join_lines(ctx, new_tables)
    if config.siblings_enabled():
        named += [
            (line, set().union(*(own(t) for t in tables)))
            for line, tables in sibling_lines(ctx, new_tables)
        ]
    named += [
        (block, _sql_tables(hit.get("sql")))
        for hit in hits
        if (block := _definition_block(hit)) is not None
    ]
    named += [(_table_block(t, matched[str(t["id"])], True), own(t)) for t in revisited]
    if not named:
        return (
            f"No matches for {query!r}. Try other words, an exact column or table "
            "name, or a broader query."
        )
    result = _cap([block for block, _ in named])
    # Only a table the model was actually shown counts as seen; one the limit
    # cut must come back in full when a later search finds it again.
    ctx.seen_tables.update(
        {str(t["id"]): t for t in new_tables if new_blocks[str(t["id"])] in result}
    )
    for block, tables in named:
        if block in result:
            for schema, name in tables:
                ctx.show_table(schema, name)
    return result
