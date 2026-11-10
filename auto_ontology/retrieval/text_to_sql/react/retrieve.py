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

1. **Search by meaning** — :func:`_get_candidates_information`, the vector
   search plus graph enrichment candidate retrieval already uses.
2. **Search by name** — :func:`fetch_global_search`, the catalog's substring
   search, which finds codes an embedding misses (``CDSCode``, ``T-BIL``).
3. **Expand one step along the graph** — each hit's tables with their columns,
   stored values and semantic FKs (:func:`fetch_tables_by_ids`), the join route
   from every new table to the tables earlier calls already showed
   (:func:`find_join_path`), and the near-copies of every new table
   (:mod:`.siblings`).

Terms and column attributes are not searched: a table's search text carries
its term and a column's carries its attribute, so the table or column itself
is the hit.

The result is rendered as text and capped in size. The same query against the
same catalog returns the same text.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from auto_ontology.catalog.constants import Labels
from auto_ontology.dal.attributes import find_join_path
from auto_ontology.dal.candidates import expand_info
from auto_ontology.dal.datasources import (
    fetch_tables_by_ids,
    find_column_id_by_table_and_name,
)
from auto_ontology.dal.search import fetch_global_search, search_tokens
from auto_ontology.retrieval.data_access.candidates import _get_candidates_information
from auto_ontology.retrieval.text_to_sql.react import config
from auto_ontology.retrieval.text_to_sql.react.context import ToolContext
from auto_ontology.retrieval.text_to_sql.react.siblings import sibling_lines
from auto_ontology.semantic.constants import LABEL_SQL_ATTRIBUTE

logger = logging.getLogger(__name__)

#: What the model may narrow a search to, and the graph labels each covers.
KIND_LABELS: dict[str, list[str]] = {
    "table": [Labels.TABLE],
    "column": [Labels.COLUMN],
    "definition": [LABEL_SQL_ATTRIBUTE],
    "analysis": [Labels.CUSTOM_ANALYSIS],
}

#: Hits kept per label from the meaning search. Columns get the most because
#: they are what a query is written from; definitions are few and long.
_PER_LABEL_K: dict[str, int] = {
    Labels.TABLE: 8,
    Labels.COLUMN: 15,
    LABEL_SQL_ATTRIBUTE: 3,
    Labels.CUSTOM_ANALYSIS: 3,
}

_NAME_SEARCH_LIMIT = 10
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


#: Labels embedded in the data collection rather than the semantic one.
_DATA_LABELS = (Labels.TABLE, Labels.COLUMN)


def _meaning_hits(ctx: ToolContext, query: str, labels: list[str]) -> list[dict]:
    """Vector hits for *labels*, each label searched in the collection it lives in."""
    searches = [
        (
            ctx.data_retriever or ctx.retriever,
            [lb for lb in labels if lb in _DATA_LABELS],
        ),
        (ctx.retriever, [lb for lb in labels if lb not in _DATA_LABELS]),
    ]
    hits: list[dict] = []
    for retriever, group in searches:
        if not group or retriever is None:
            continue
        hits += _get_candidates_information(
            retriever,
            query,
            list_of_semantic=group,
            database_name=ctx.database_name,
            per_label_k={label: _PER_LABEL_K.get(label, 5) for label in group},
        )
    return hits


def _name_hits(query: str, labels: list[str]) -> list[dict]:
    """Substring hits on names: the whole query, then each identifier in it.

    The whole query has to match every token in one name or one description,
    which a sentence rarely does — so a word that looks copied from a schema is
    also searched alone, by name only.
    """
    object_types = set(labels)
    searches = [(search_tokens(query), True)]
    searches += [(search_tokens(word), False) for word in _identifier_words(query)]

    raw: list[dict] = []
    for tokens, include_description in searches:
        if not tokens:
            continue
        raw += fetch_global_search(
            tokens,
            object_types,
            include_description=include_description,
            limit=_NAME_SEARCH_LIMIT,
        )
    hits = [{"id": h["id"], "label": h["label"], "name": h.get("name")} for h in raw]
    extra = expand_info([{"id": h["id"], "label": h["label"]} for h in hits])
    for hit in hits:
        hit.update(extra.get(hit["id"]) or extra.get(str(hit["id"])) or {})
    return hits


def _merge_hits(meaning: list[dict], names: list[dict]) -> list[dict]:
    """Meaning hits first, then name hits neither search has already returned."""
    merged: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for hit in [*meaning, *names]:
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


def _resolve_tables(hits: list[dict]) -> tuple[list[str], dict[str, set[str]]]:
    """Tables the hits live on, in hit order, plus which columns each matched."""
    table_ids: list[str] = []
    matched: dict[str, set[str]] = {}
    for hit in hits:
        label = hit.get("label")
        if label == Labels.TABLE:
            ids = [str(hit["id"])]
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


def _join_lines(ctx: ToolContext, new_tables: list[dict]) -> list[str]:
    """Join routes from each new table to every table already in view."""
    lines: list[str] = []
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
                lines.append(
                    f"JOIN {_qualified(table)} ↔ {_qualified(other)}: "
                    + " → ".join(_hop_text(h) for h in hops)
                )
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
    hits = _merge_hits(_meaning_hits(ctx, query, labels), _name_hits(query, labels))

    table_ids, matched = _resolve_tables(hits)
    new_ids = [tid for tid in table_ids if tid not in ctx.seen_tables]
    new_ids = new_ids[: config.max_tables_per_retrieve()]
    fetched = {str(t["id"]): t for t in fetch_tables_by_ids(new_ids)}
    new_tables = [
        fetched[tid]
        for tid in new_ids
        if tid in fetched and _in_database(ctx, fetched[tid])
    ]
    revisited = [
        ctx.seen_tables[tid]
        for tid in table_ids
        if tid in ctx.seen_tables and matched.get(tid)
    ]

    definitions = [
        block for hit in hits if (block := _definition_block(hit)) is not None
    ]
    # Most useful first, since the limit cuts from the end: new tables, how they
    # join, definitions that carry SQL, then tables seen before.
    new_blocks = {
        str(t["id"]): _table_block(t, matched.get(str(t["id"]), set()), False)
        for t in new_tables
    }
    blocks = list(new_blocks.values())
    blocks += _join_lines(ctx, new_tables)
    if config.siblings_enabled():
        blocks += sibling_lines(ctx, new_tables)
    blocks += definitions
    blocks += [_table_block(t, matched[str(t["id"])], True) for t in revisited]
    if not blocks:
        return (
            f"No matches for {query!r}. Try other words, an exact column or table "
            "name, or a broader query."
        )
    result = _cap(blocks)
    # Only a table the model was actually shown counts as seen; one the limit
    # cut must come back in full when a later search finds it again.
    ctx.seen_tables.update(
        {str(t["id"]): t for t in new_tables if new_blocks[str(t["id"])] in result}
    )
    return result
