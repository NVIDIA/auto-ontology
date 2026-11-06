# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared prompt formatters for text-to-SQL agents."""

from __future__ import annotations

from typing import Any

from auto_ontology.utils.sample_values import stringify_sample_values


def _format_sample_values(raw: Any) -> str:
    """Render sample values as ``a, b, c``, or empty when there are none.

    Matches how every other model-facing renderer spells them out (see
    ``auto_ontology.semantic.embed``); interpolating the list itself would leak Python
    repr punctuation into the prompt.
    """
    values = stringify_sample_values(raw)
    if not values:
        return ""
    return ", ".join(values)


def _plain_cell(value: Any) -> str:
    """One pipe-table cell: no newlines, and no ``|`` that would add a column."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        parts = [_plain_cell(item) for item in value]
        return ", ".join(part for part in parts if part)
    return " ".join(str(value).replace("|", "/").split())


def _pipe_table(headers: list[str], rows: list[dict[str, str]]) -> str:
    """Render ``header 1 | header 2`` then one ``|``-separated row per record.

    A header is omitted when every cell under it is empty, so a table with no
    descriptions does not pay for a description column.
    """
    active = [
        header for header in headers if any(row.get(header, "").strip() for row in rows)
    ]
    if not active:
        return ""
    lines = [" | ".join(active)]
    for row in rows:
        lines.append(" | ".join(row.get(header, "") for header in active))
    return "\n".join(lines)


def _format_column_for_prompt(column: dict[str, Any], *, indent: str = "") -> str:
    """Render one catalog column consistently wherever schema context appears."""
    name = column.get("name", "UNKNOWN")
    data_type = column.get("data_type", "UNKNOWN")
    description = column.get("description", "")
    sample_values = _format_sample_values(column.get("sample_values"))

    line = f"{indent}- {name} ({data_type})"
    if description:
        line += f" - {description}"
    if sample_values:
        if "json" in (data_type or "").lower():
            line += (
                " | available JSONB keys"
                " (dot = nesting level, use as"
                f" ->>'key' or ->'container'->>'leaf'): {sample_values}"
            )
        else:
            line += f" | sample values: {sample_values}"
    notation = column.get("format")
    if notation and "format:" not in (description or "").lower():
        line += f" | format: {notation}"
    nullability = column.get("is_nullable")
    if nullability is True:
        line += " | is_nullable: true"
    elif nullability is False:
        line += " | is_nullable: false"
    else:
        line += " | is_nullable: unknown"
    semantic_fk = column.get("semantic_fk")
    if semantic_fk:
        line += f" | foreign_key: {_plain_cell(semantic_fk)}"
    return line


# Dialects whose connection does *not* bind the catalog, so a name has to carry
# it explicitly. These are exactly the connectors that override
# ``SQLDatabase.qualify`` to prepend it (``auto_ontology.connectors.kyuubi``,
# ``auto_ontology.connectors.trino``); everywhere else the session is already scoped to
# the database and a two-part name resolves.
_CATALOG_QUALIFIED_DIALECTS = frozenset({"spark", "trino"})


def qualify_table(
    database_name: str,
    schema_name: str,
    table_name: str,
    dialect: str | None = None,
) -> str:
    """Build the qualified identifier the model is expected to copy verbatim.

    Two-level dialects (MySQL/MariaDB) have no schema namespace: the catalog
    reports ``TABLE_SCHEMA`` as the database itself, so naively joining all
    three parts yields ``db.db.table``, which is a syntax error. The duplicate
    is collapsed for those, and for engines whose connection binds the database
    so that ``schema.table`` still resolves.

    It must *not* be collapsed on a dialect in
    :data:`_CATALOG_QUALIFIED_DIALECTS`. There the catalog and the schema are
    separate namespaces that may legitimately share a name — a Kyuubi catalog
    ``lakehouse`` holding a schema ``lakehouse`` — and the session is left on
    the engine's own default catalog (``spark_catalog``), not the bound one.
    So collapsing emits ``lakehouse.events``, which Spark reads as schema
    ``lakehouse`` under ``spark_catalog`` and rejects with
    TABLE_OR_VIEW_NOT_FOUND. Scoping the session instead is not an option:
    Spark rejects ``USE CATALOG``, and ``SET CATALOG`` leaves
    ``current_schema()`` empty so the two-part name still does not resolve.
    """
    parts = [database_name, schema_name, table_name]
    collapsible = (dialect or "").lower() not in _CATALOG_QUALIFIED_DIALECTS
    if (
        collapsible
        and database_name
        and schema_name
        and database_name.lower() == schema_name.lower()
    ):
        parts = [schema_name, table_name]
    return ".".join(part for part in parts if part)


def _hop_column(
    hop: dict,
    side: str,
    target_db: str | None = None,
    dialect: str | None = None,
) -> str:
    """Format a join-path endpoint, qualified as :func:`qualify_table` would.

    The hop's own database wins over *target_db*, so a path that crosses
    databases names each side correctly; only the bridge-table builder in
    ``auto_ontology.dal.attributes`` omits it, and there *target_db* is the fallback.
    """
    database = hop.get(f"{side}_database") or target_db or ""
    schema = hop.get(f"{side}_schema", "")
    table = hop.get(f"{side}_table", "")
    column = hop.get(f"{side}_column", "")
    prefix = qualify_table(database, schema, table, dialect)
    return f"{prefix}.{column}"


def hop_column_pairs(hop: dict) -> list[dict]:
    """``{source_column, target_column}`` for every column a hop equates.

    One pair for an ordinary hop; a composite key's hop lists them all in
    ``column_pairs``.
    """
    pairs = hop.get("column_pairs")
    if isinstance(pairs, list) and pairs:
        return pairs
    return [
        {
            "source_column": hop.get("source_column", ""),
            "target_column": hop.get("target_column", ""),
        }
    ]


def _hop_condition(
    hop: dict, target_db: str | None = None, dialect: str | None = None
) -> str:
    """``a.x = b.x``, or ``a.x = b.x AND a.y = b.y`` for a composite key."""
    return " AND ".join(
        f"{_hop_column({**hop, **pair}, 'source', target_db, dialect)} = "
        f"{_hop_column({**hop, **pair}, 'target', target_db, dialect)}"
        for pair in hop_column_pairs(hop)
    )


CONCEPT_JOIN_RULES = (
    "A concept is one real-world thing. Its PK columns identify it on each table "
    "that holds it; its FK columns reference it from other tables. Within one "
    "concept:\n"
    "- Any PK may be joined to any other PK, and any FK to any PK.\n"
    "- Never join two FKs to each other. Tables that both reference a thing are "
    "related only through it: join each of them to one of its PK tables instead.\n"
    "- Columns are equated position by position. A key of several columns joins "
    "on all of them, combined with AND, in the order listed.\n"
    "- A table with several FKs to one concept references it in different roles. "
    "Use the one the question means, or the one marked [default] when it does "
    "not say."
)


def _concept_side(
    entry: dict, target_db: str | None, dialect: str | None, *, default: bool = False
) -> str:
    table = qualify_table(
        entry.get("database_name") or target_db or "",
        entry.get("schema_name", ""),
        entry.get("table_name", ""),
        dialect,
    )
    marker = " [default]" if default else ""
    return f"{table} ({', '.join(entry.get('columns') or [])}){marker}"


def format_concepts_for_prompt(
    concepts: list[dict],
    target_db: str | None = None,
    dialect: str | None = None,
) -> str:
    """A CONCEPTS section: each concept's PK and FK columns and the join rules.

    *concepts* is what :func:`auto_ontology.dal.concepts.fetch_concepts_for_tables`
    returns. Names are qualified as :func:`format_tables_for_prompt` does.
    ``""`` when there are none.
    """
    if not concepts:
        return ""
    lines = ["CONCEPTS (join keys from the semantic model):", CONCEPT_JOIN_RULES, ""]
    for concept in concepts:
        header = f"- {_plain_cell(concept.get('name'))}"
        if concept.get("description"):
            header += f": {_plain_cell(concept['description'])}"
        lines.append(header)
        keys = concept.get("keys") or []
        lines.append(
            "    PK: "
            + "; ".join(_concept_side(key, target_db, dialect) for key in keys)
        )
        references = concept.get("references") or []
        per_table: dict[str, int] = {}
        for reference in references:
            per_table[reference["table_id"]] = (
                per_table.get(reference["table_id"], 0) + 1
            )
        if references:
            lines.append(
                "    FK: "
                + "; ".join(
                    _concept_side(
                        reference,
                        target_db,
                        dialect,
                        default=per_table[reference["table_id"]] > 1
                        and bool(reference.get("use_in_join_paths")),
                    )
                    for reference in references
                )
            )
    return "\n".join(lines)


def _concept_roles(concepts: list[dict]) -> dict[tuple[str, str, str], set]:
    """``(schema, table, column)`` -> ``{(concept id, "pk" | "fk")}``."""
    roles: dict[tuple[str, str, str], set] = {}
    for concept in concepts:
        for role, entries in (
            ("pk", concept.get("keys") or []),
            ("fk", concept.get("references") or []),
        ):
            for entry in entries:
                for column in entry.get("columns") or []:
                    location = (
                        str(entry.get("schema_name") or "").casefold(),
                        str(entry.get("table_name") or "").casefold(),
                        str(column).casefold(),
                    )
                    roles.setdefault(location, set()).add((concept["id"], role))
    return roles


def _covered_by_concept(hop: dict, roles: dict[tuple[str, str, str], set]) -> bool:
    """Whether some concept already allows every column pair the hop equates."""

    def side(pair: dict, name: str) -> set:
        return roles.get(
            (
                str(hop.get(f"{name}_schema") or "").casefold(),
                str(hop.get(f"{name}_table") or "").casefold(),
                str(pair.get(f"{name}_column") or "").casefold(),
            ),
            set(),
        )

    allowed: set | None = None
    for pair in hop_column_pairs(hop):
        source, target = side(pair, "source"), side(pair, "target")
        here = {
            concept_id
            for concept_id, role in source
            for other_id, other_role in target
            if concept_id == other_id and "pk" in (role, other_role)
        }
        allowed = here if allowed is None else allowed & here
        if not allowed:
            return False
    return bool(allowed)


def format_other_joins_for_prompt(
    attribute_join_paths: list[dict],
    concepts: list[dict],
    target_db: str | None = None,
    dialect: str | None = None,
) -> str:
    """Verified join conditions no concept covers, each listed once.

    ``""`` when every hop is covered by :func:`format_concepts_for_prompt`.
    """
    roles = _concept_roles(concepts)
    conditions = list(
        dict.fromkeys(
            _hop_condition(hop, target_db, dialect)
            for entry in attribute_join_paths
            for hop in entry.get("path") or []
            if not _covered_by_concept(hop, roles)
        )
    )
    if not conditions:
        return ""
    heading = (
        "OTHER JOINS (verified, outside any concept; use these exact conditions):"
        if concepts
        else "JOINS (verified; use these exact conditions):"
    )
    return "\n".join([heading, *(f"  {c}" for c in conditions)])


def format_semantic_context(
    primary_attribute: dict | None,
    attribute_join_paths: list[dict],
    target_db: str | None = None,
    dialect: str | None = None,
    concepts: list[dict] | None = None,
) -> str:
    """Format the semantic anchor, the concepts and other verified joins.

    Joins inside a concept are described by its PK and FK columns rather than
    listed pair by pair; only hops no concept covers are spelled out. Names
    are qualified exactly as :func:`format_tables_for_prompt` does — a name
    spelled differently here than in the schema context is one the model may
    copy into SQL, and on a catalog-qualified engine dropping the catalog
    makes it unresolvable.
    """
    blocks: list[str] = []
    if primary_attribute:
        anchor_schema = primary_attribute.get("schema_name", "")
        anchor_table = primary_attribute.get("table_name", "")
        anchor_col = primary_attribute.get("col_name", "")
        anchor_name = primary_attribute.get("attr_name", "")
        anchor_database = primary_attribute.get("database_name") or target_db or ""
        anchor_full = qualify_table(
            anchor_database, anchor_schema, anchor_table, dialect
        )
        # Only present on attrs (re-)ingested since this field was added — older
        # rows just omit the tag.
        anchor_datatype = primary_attribute.get("datatype") or ""
        anchor_datatype_tag = f" [{anchor_datatype}]" if anchor_datatype else ""
        blocks.append(
            "\n".join(
                [
                    "SEMANTIC HINT — likely starting table (use as a strong hint, "
                    "not a mandate):",
                    f"  Table: {anchor_full}",
                    f"  Column: {anchor_col}  ({anchor_name}){anchor_datatype_tag}",
                ]
            )
        )
    blocks.append(format_concepts_for_prompt(concepts or [], target_db, dialect))
    blocks.append(
        format_other_joins_for_prompt(
            attribute_join_paths, concepts or [], target_db, dialect
        )
    )
    return "\n\n".join(block for block in blocks if block)


def format_important_columns_for_prompt(
    primary_attribute: dict | None,
    attribute_join_paths: list[dict],
    tables: list[dict],
    target_db: str | None = None,
    dialect: str | None = None,
) -> str:
    """Render columns selected by semantic retrieval with full catalog details.

    ``primary_attribute`` and named entries in ``attribute_join_paths`` identify
    columns whose ColumnAttributes matched extracted question entities. Their
    owning tables also appear in the ordinary schema section; this deliberately
    repeats the matched columns so the SQL model can distinguish them from the
    rest of a potentially wide table.
    """
    entries = [
        entry
        for entry in [primary_attribute, *attribute_join_paths]
        if entry
        and entry.get("table_name")
        and entry.get("col_name")
        and entry.get("attr_name")
    ]
    if not entries:
        return "No semantically matched columns."

    lines: list[str] = []
    for entry in entries:
        database = entry.get("database_name") or target_db or ""
        schema = entry.get("schema_name") or ""
        table_name = entry.get("table_name") or ""
        column_name = entry.get("col_name") or ""

        matching_table = next(
            (
                table
                for table in tables
                if str(table.get("name") or "").casefold() == table_name.casefold()
                and (
                    not schema
                    or str(table.get("schema_name") or "").casefold()
                    == schema.casefold()
                )
                and (
                    not database
                    or not table.get("database_name")
                    or str(table.get("database_name") or "").casefold()
                    == database.casefold()
                )
            ),
            None,
        )
        # An attribute's table may not be among the tables in scope. Do not
        # introduce a column from a table absent from
        # AVAILABLE TABLES, where the model is explicitly forbidden to use it.
        if matching_table is None:
            continue
        catalog_column = next(
            (
                column
                for column in (matching_table or {}).get("columns", [])
                if isinstance(column, dict)
                and str(column.get("name") or "").casefold() == column_name.casefold()
            ),
            None,
        )
        column = catalog_column or {
            "name": column_name,
            "data_type": entry.get("datatype") or "UNKNOWN",
        }
        full_table = qualify_table(database, schema, table_name, dialect)
        lines.append(f"Semantic match: {entry['attr_name']}")
        lines.append(f"  Table: {full_table}")
        lines.append(_format_column_for_prompt(column, indent="  "))

    return "\n".join(lines) or "No semantically matched columns."


_TABLE_HEADERS = ["table", "label", "description", "primary key"]
_COLUMN_HEADERS = [
    "column",
    "type",
    "is_nullable",
    "foreign key",
    "description",
    "sample values",
    "format",
]
_JSONB_SAMPLE_NOTE = (
    "JSONB sample values are keys (dot = nesting level, use as "
    "->>'key' or ->'container'->>'leaf')."
)


def _column_row(column: Any) -> dict[str, str]:
    if not isinstance(column, dict):
        return {"column": _plain_cell(column)}
    description = _plain_cell(column.get("description"))
    notation = _plain_cell(column.get("format"))
    if notation and "format:" in description.lower():
        notation = ""
    nullability = column.get("is_nullable")
    if nullability is True:
        nullable = "true"
    elif nullability is False:
        nullable = "false"
    else:
        nullable = "unknown"
    return {
        "column": _plain_cell(column.get("name", "UNKNOWN")),
        "type": _plain_cell(column.get("data_type", "UNKNOWN")),
        "is_nullable": nullable,
        "foreign key": _plain_cell(column.get("semantic_fk")),
        "description": description,
        "sample values": _plain_cell(
            _format_sample_values(column.get("sample_values"))
        ),
        "format": notation,
    }


_ENTITY_MATCH_NOTE = (
    "Tables are ordered by how many of the question's entities they match, then "
    "by how closely. Several tables may match the same entity; choose by what "
    "the question asks, not by position."
)


def _format_entity_matches(matches: Any) -> str:
    """One ``matches:`` line naming each entity this table matches, and through which column."""
    if not isinstance(matches, list) or not matches:
        return ""
    parts = []
    for match in matches:
        if not isinstance(match, dict) or not match.get("entity"):
            continue
        part = f'"{_plain_cell(match["entity"])}"'
        if match.get("column"):
            part += f" via {_plain_cell(match['column'])}"
        parts.append(part)
    return f"matches: {'; '.join(parts)}" if parts else ""


def _full_table_name(table: dict, target_db: str | None, dialect: str | None) -> str:
    return qualify_table(
        table.get("database_name") or target_db or "",
        table.get("schema_name", ""),
        table.get("name", "UNKNOWN"),
        dialect,
    )


def format_entity_tables_for_prompt(
    tables: list[dict],
    entities: list[str] | None = None,
    target_db: str | None = None,
    dialect: str | None = None,
) -> str:
    """An ENTITIES section: each question entity with its matching tables, in rank order.

    Built from the ``entity_matches`` that
    :func:`~auto_ontology.retrieval.text_to_sql.table_ranking.rank_tables_by_entity`
    puts on *tables*, and naming each table exactly as
    :func:`format_tables_for_prompt` does. Entities follow *entities* order;
    any others follow in the order their tables appear. ``""`` when no table
    carries a match.
    """
    per_entity: dict[str, list[tuple[int, int, str]]] = {}
    for position, table in enumerate(tables or []):
        matches = table.get("entity_matches")
        if not isinstance(matches, list):
            continue
        name = _full_table_name(table, target_db, dialect)
        for match in matches:
            if not isinstance(match, dict) or not match.get("entity"):
                continue
            label = name
            if match.get("column"):
                label += f" (via {_plain_cell(match['column'])})"
            rank = match.get("rank")
            per_entity.setdefault(match["entity"], []).append(
                (rank if isinstance(rank, int) else len(tables), position, label)
            )
    if not per_entity:
        return ""

    ordered = [e for e in entities or [] if e in per_entity]
    ordered += [e for e in per_entity if e not in ordered]
    lines = [
        "ENTITIES (each entity in the question and the tables above with a "
        "matching column; tables matching more of the question's entities come "
        "first, then closer matches):"
    ]
    for entity in ordered:
        ranked = ", ".join(label for *_, label in sorted(per_entity[entity]))
        lines.append(f'- "{_plain_cell(entity)}": {ranked}')
    return "\n".join(lines)


def format_tables_for_prompt(
    tables: list[dict],
    target_db: str | None = None,
    dialect: str | None = None,
) -> str:
    """Format tables and their columns as schema context for a prompt.

    *dialect* only decides how each table name is qualified; see
    :func:`qualify_table`. Omitting it keeps the two-part form, which is wrong
    for a catalog-qualified engine, so callers that have a connector should
    pass its dialect.

    *target_db* stands in for a table that carries no ``database_name`` of its
    own, which legacy catalog metadata does not (see
    ``connector_routing.resolve_connector_from_tables``). Without it such a
    table renders unqualified here while :func:`format_semantic_context` names
    the same table with its catalog — and on a catalog-qualified engine the
    short one does not resolve.
    """
    if not tables:
        return "No tables available"

    def full_name_of(table: dict) -> str:
        return _full_table_name(table, target_db, dialect)

    formatted_tables = []
    if any(table.get("entity_matches") for table in tables):
        formatted_tables.append(_ENTITY_MATCH_NOTE)
    for table in tables:
        table_name = table.get("name", "UNKNOWN")
        table_label = table.get("label", "")
        full_name = full_name_of(table)
        blocks = [
            _pipe_table(
                _TABLE_HEADERS,
                [
                    {
                        "table": _plain_cell(full_name),
                        "label": _plain_cell(
                            table_label if table_label != table_name else ""
                        ),
                        "description": _plain_cell(table.get("description")),
                        "primary key": _plain_cell(table.get("pk")),
                    }
                ],
            )
        ]
        entity_line = _format_entity_matches(table.get("entity_matches"))
        if entity_line:
            blocks.append(entity_line)

        columns = table.get("columns")
        if not isinstance(columns, list):
            columns = []
        if columns:
            column_rows = [_column_row(column) for column in columns]
            blocks.append(_pipe_table(_COLUMN_HEADERS, column_rows))
            if any("json" in row["type"].lower() for row in column_rows):
                blocks.append(_JSONB_SAMPLE_NOTE)

        formatted_tables.append("\n".join(block for block in blocks if block))

    return "\n\n".join(formatted_tables)


def format_used_schema_for_prompt(
    tables: list[dict],
    used_tables: list[str],
    used_columns: list[str],
    target_db: str | None = None,
    dialect: str | None = None,
) -> str:
    """Enrich only the parsed table and column references used by the SQL."""

    def tail(value: str, size: int) -> tuple[str, ...]:
        return tuple(part.casefold() for part in value.split(".")[-size:])

    def matching_table(used_name: str) -> dict | None:
        used_schema, used_table = tail(used_name, 2)
        return next(
            (
                table
                for table in tables
                if str(table.get("name") or "").casefold() == used_table
                and (
                    not table.get("schema_name")
                    or str(table.get("schema_name") or "").casefold() == used_schema
                )
            ),
            None,
        )

    table_lines: list[str] = []
    for used_name in used_tables:
        table = matching_table(used_name)
        if table is None:
            table_lines.append(f"- {used_name}")
            continue
        full_name = qualify_table(
            table.get("database_name") or target_db or "",
            table.get("schema_name") or "",
            table.get("name") or "",
            dialect,
        )
        line = f"- {full_name}"
        label = table.get("label")
        if label and label != table.get("name"):
            line += f" | label: {label}"
        if table.get("description"):
            line += f" | description: {table['description']}"
        if table.get("pk"):
            line += f" | primary key: {table['pk']}"
        table_lines.append(line)

    column_lines: list[str] = []
    for used_name in used_columns:
        used_schema, used_table, used_column = tail(used_name, 3)
        table = matching_table(f"{used_schema}.{used_table}")
        if table is None:
            column_lines.append(f"- {used_name}")
            continue
        columns = table.get("columns")
        if not isinstance(columns, list):
            columns = []
        column = next(
            (
                candidate
                for candidate in columns
                if isinstance(candidate, dict)
                and str(candidate.get("name") or "").casefold() == used_column
            ),
            None,
        )
        if column is None:
            column_lines.append(f"- {used_name}")
            continue
        full_table = qualify_table(
            table.get("database_name") or target_db or "",
            table.get("schema_name") or "",
            table.get("name") or "",
            dialect,
        )
        rendered = _format_column_for_prompt(column)
        rendered = rendered.replace(
            f"- {column.get('name', 'UNKNOWN')}",
            f"- {full_table}.{column.get('name', 'UNKNOWN')}",
            1,
        )
        column_lines.append(rendered)

    return (
        "TABLES USED IN SQL:\n"
        + ("\n".join(table_lines) or "- None resolved")
        + "\n\nCOLUMNS USED IN SQL:\n"
        + ("\n".join(column_lines) or "- None resolved")
    )
