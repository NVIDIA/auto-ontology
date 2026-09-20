"""Shared prompt formatters for text-to-SQL agents."""

from __future__ import annotations

from typing import Any

from gsf.utils.sample_values import stringify_sample_values


def _format_sample_values(raw: Any) -> str:
    """Render sample values as ``a, b, c``, or empty when there are none.

    Matches how every other model-facing renderer spells them out (see
    ``gsf.semantic.embed``); interpolating the list itself would leak Python
    repr punctuation into the prompt.
    """
    values = stringify_sample_values(raw)
    if not values:
        return ""
    return ", ".join(values)


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
    return line


# Dialects whose connection does *not* bind the catalog, so a name has to carry
# it explicitly. These are exactly the connectors that override
# ``SQLDatabase.qualify`` to prepend it (``gsf.connectors.kyuubi``,
# ``gsf.connectors.trino``); everywhere else the session is already scoped to
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
    ``gsf.dal.attributes`` omits it, and there *target_db* is the fallback.
    """
    database = hop.get(f"{side}_database") or target_db or ""
    schema = hop.get(f"{side}_schema", "")
    table = hop.get(f"{side}_table", "")
    column = hop.get(f"{side}_column", "")
    prefix = qualify_table(database, schema, table, dialect)
    return f"{prefix}.{column}"


def format_semantic_context(
    primary_attribute: dict,
    attribute_join_paths: list[dict],
    target_db: str | None = None,
    dialect: str | None = None,
) -> str:
    """Format the semantic anchor and verified optional join paths for a prompt.

    Names are qualified exactly as :func:`format_tables_for_prompt` does. The
    prompt tells the model to copy a selected path's verified join conditions,
    so a name spelled differently here than in the schema context is one the
    model may copy into SQL — on a catalog-qualified engine, dropping the catalog
    makes it unresolvable.
    """
    anchor_schema = primary_attribute.get("schema_name", "")
    anchor_table = primary_attribute.get("table_name", "")
    anchor_col = primary_attribute.get("col_name", "")
    anchor_name = primary_attribute.get("attr_name", "")
    anchor_database = primary_attribute.get("database_name") or target_db or ""
    anchor_full = qualify_table(anchor_database, anchor_schema, anchor_table, dialect)
    # Only present on attrs (re-)ingested since this field was added — older
    # rows just omit the tag.
    anchor_datatype = primary_attribute.get("datatype") or ""
    anchor_datatype_tag = f" [{anchor_datatype}]" if anchor_datatype else ""

    lines: list[str] = [
        "SEMANTIC HINT — likely starting table (use as a strong hint, not a mandate):",
        f"  Table: {anchor_full}",
        f"  Column: {anchor_col}  ({anchor_name}){anchor_datatype_tag}",
    ]

    if attribute_join_paths:
        lines.append("")
        lines.append(
            "JOIN PATHS (VERIFIED OPTIONS — derived from the semantic model). "
            "These paths show how available tables can join; they are not a plan "
            "and most may be irrelevant to this question. Select only paths needed "
            "to reach tables that contribute to the answer. Never add a path merely "
            "because it is listed. When a path is needed, use its exact join "
            "conditions and only the required hops:"
        )
        for entry in attribute_join_paths:
            attr_name = entry.get("attr_name", "")
            col_name = entry.get("col_name", "")
            schema = entry.get("schema_name", "")
            table = entry.get("table_name", "")
            datatype = entry.get("datatype") or ""
            datatype_tag = f" [{datatype}]" if datatype else ""
            # Structural entries (hub-sibling / bridge-table reconciliation)
            # carry only "path" — no named attribute they resolve to. Render
            # a generic label instead of a blank "  : ." header line.
            if attr_name or col_name or table:
                database = entry.get("database_name") or target_db or ""
                full_table = qualify_table(database, schema, table, dialect)
                lines.append(f"  {attr_name}: {full_table}.{col_name}{datatype_tag}")
            else:
                lines.append("  (structural bridge — connects tables kept above)")
            path = entry.get("path") or []
            if path:
                lines.append("    Join path:")
                if len(path) == 1:
                    left = _hop_column(path[0], "source", target_db, dialect)
                    right = _hop_column(path[0], "target", target_db, dialect)
                    lines.append(f"      {left} = {right}")
                else:
                    for cur, nxt in zip(path, path[1:]):
                        left = _hop_column(cur, "target", target_db, dialect)
                        right = _hop_column(nxt, "source", target_db, dialect)
                        lines.append(f"      {left} = {right}")

    return "\n".join(lines)


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
        # The relevance filter may have removed an attribute's table after join
        # paths were built. Do not reintroduce a column from a table absent from
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

    formatted_tables = []
    for table in tables:
        table_parts = []

        table_name = table.get("name", "UNKNOWN")
        table_label = table.get("label", "")
        table_description = table.get("description", "")
        database_name = table.get("database_name") or target_db or ""
        schema_name = table.get("schema_name", "")

        full_name = qualify_table(database_name, schema_name, table_name, dialect)

        table_parts.append(f"TABLE: {full_name}")
        if table_label and table_label != table_name:
            table_parts.append(f"  Label: {table_label}")
        if table_description:
            table_parts.append(f"  Description: {table_description}")
        if table.get("pk"):
            table_parts.append(f"  Primary Key: {table['pk']}")

        columns = table.get("columns")
        if not isinstance(columns, list):
            columns = []
        if columns:
            table_parts.append(
                "  AVAILABLE COLUMNS (only use these columns for this table):"
            )
            for col in columns:
                if isinstance(col, dict):
                    table_parts.append(_format_column_for_prompt(col, indent="    "))
                elif isinstance(col, str):
                    table_parts.append(f"    - {col}")
                else:
                    table_parts.append(f"    - {str(col)}")

        formatted_tables.append("\n".join(table_parts))

    return "\n\n".join(formatted_tables)
