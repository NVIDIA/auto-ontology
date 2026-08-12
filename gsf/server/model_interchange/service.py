# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Orchestration for native GSF and Apache Ossie model YAML export/import."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Mapping
from typing import Any

import yaml
from ossie_gsf import GSFConversionError, convert_gsf_to_ossie, convert_ossie_to_gsf

from gsf.connectors import get_connectors
from gsf.dal import model_interchange as dal
from gsf.dal.connections import list_connections
from gsf.server.model_interchange.embed import (
    ImportEmbedBuffer,
    flush_import_embeddings,
)
from gsf.server.model_interchange.schemas import (
    ExportRequest,
    GsfModelDocument,
    ModelFormat,
)

logger = logging.getLogger(__name__)


def _dialect_by_database_name() -> dict[str, str]:
    """Resolve SQL dialect strings keyed by catalog database name."""
    dialects: dict[str, str] = {}
    for connector in get_connectors():
        database_name = getattr(connector, "database_name", None)
        dialect = getattr(connector, "dialect", None)
        if database_name and dialect:
            dialects[str(database_name)] = str(dialect)

    for connection in list_connections():
        database_name = str(
            connection.get("database")
            or connection.get("database_name")
            or connection.get("name")
            or "",
        )
        if not database_name or database_name in dialects:
            continue
        dialect = connection.get("dialect") or connection.get("type") or ""
        dialects[database_name] = str(dialect)
    return dialects


def detect_model_format(payload: Mapping[str, Any]) -> ModelFormat:
    """Tell an Apache Ossie document apart from a native GSF one.

    The two vocabularies do not overlap at the root: Ossie holds its models
    under ``semantic_model``, GSF under ``data_layer``/``semantic_layer``.
    """
    return ModelFormat.OSSIE if "semantic_model" in payload else ModelFormat.GSF


def _table_id_by_column_id(document: GsfModelDocument) -> dict[str, str]:
    """Map every catalog column id to the id of the table that owns it."""
    return {
        column.id: table.id
        for database in document.data_layer.databases
        for schema in database.schemas
        for table in schema.tables
        for column in table.columns
    }


def _project_terms_onto_one_table(document: GsfModelDocument) -> GsfModelDocument:
    """Narrow every term down to a single represented table.

    An Ossie dataset is backed by exactly one table, while a GSF term may
    represent several. Until the converter can split such a term into one
    dataset per table, the export keeps the table carrying most of the term's
    column attributes and drops the attributes belonging to the others, so
    that no field ends up referencing a column the dataset does not have.
    """
    table_of_column = _table_id_by_column_id(document)
    projected = document.model_copy(deep=True)
    for term in projected.semantic_layer.terms:
        if len(term.represents) <= 1:
            continue
        attributes_per_table = Counter(
            table_of_column.get(attribute.column_id, "")
            for attribute in term.columns_attributes
        )
        kept_table_id = max(term.represents, key=lambda t: attributes_per_table[t])
        dropped = [t for t in term.represents if t != kept_table_id]
        term.represents = [kept_table_id]
        term.columns_attributes = [
            attribute
            for attribute in term.columns_attributes
            if table_of_column.get(attribute.column_id) == kept_table_id
        ]
        logger.warning(
            "Term %r represents %d tables; exporting only table %s to Ossie "
            "and dropping %s",
            term.name,
            len(dropped) + 1,
            kept_table_id,
            ", ".join(dropped),
        )

    kept_attribute_ids = {
        attribute.id
        for term in projected.semantic_layer.terms
        for attribute in term.columns_attributes
    }
    projected.semantic_layer.semantic_fks = [
        semantic_fk
        for semantic_fk in projected.semantic_layer.semantic_fks
        if semantic_fk.column_attribute_id in kept_attribute_ids
    ]
    return projected


def _dump_yaml(document: GsfModelDocument) -> str:
    return yaml.safe_dump(
        document.model_dump(mode="python"),
        sort_keys=False,
        default_flow_style=False,
    )


def _validate_ossie_metric_names(document: GsfModelDocument) -> None:
    """Reject custom analyses that Ossie's global metric namespace cannot hold."""
    counts = Counter(
        analysis.name for analysis in document.semantic_layer.custom_analyses
    )
    duplicates = [
        f"{name!r} ({count})" for name, count in counts.items() if name and count > 1
    ]
    if duplicates:
        raise GSFConversionError(
            "Cannot export custom analyses with duplicate names as Apache Ossie "
            f"metrics: {', '.join(duplicates)}"
        )


def export_model(request: ExportRequest) -> str:
    """Export the scoped model document as a YAML string in *request.format*."""
    database_ids = request.databases
    dal.validate_database_ids(database_ids)
    rows = dal.fetch_export_rows(database_ids)
    document = dal.assemble_export_document(
        rows,
        dialect_by_db_name=_dialect_by_database_name(),
        sql_column_resolver=dal.resolve_sql_column_ids,
    )
    if request.format is ModelFormat.OSSIE:
        _validate_ossie_metric_names(document)
        return convert_gsf_to_ossie(_dump_yaml(_project_terms_onto_one_table(document)))
    return _dump_yaml(document)


def import_model(
    yaml_text: str,
    *,
    replace: bool = True,
    embed: bool = True,
) -> dict[str, Any]:
    """Validate and apply a YAML model document in either supported format."""
    payload = yaml.safe_load(yaml_text)
    if not isinstance(payload, dict):
        raise ValueError("YAML document must deserialize to a mapping")
    source_format = detect_model_format(payload)
    if source_format is ModelFormat.OSSIE:
        payload = yaml.safe_load(convert_ossie_to_gsf(yaml_text))
    document = GsfModelDocument.model_validate(payload)
    embed_buffer = ImportEmbedBuffer() if embed else None
    summary = dal.apply_import_model(
        document,
        replace=replace,
        embed_buffer=embed_buffer,
    )
    summary["format"] = source_format.value
    if embed and embed_buffer is not None:
        summary["embeddings"] = flush_import_embeddings(embed_buffer)
    return summary
