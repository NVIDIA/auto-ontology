# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic models for the Auto Ontology model interchange YAML document."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ModelFormat(str, Enum):
    """Serialisation dialect of an exchanged model document."""

    AUTO_ONTOLOGY = "auto_ontology"
    OSSIE = "ossie"


class ExportRequest(BaseModel):
    """Export scope: empty ``databases`` exports every catalog database."""

    databases: list[str] = Field(default_factory=list)
    format: ModelFormat = ModelFormat.AUTO_ONTOLOGY


class ImportRequest(BaseModel):
    """Optional import flags when YAML is supplied as raw body text."""

    replace: bool = True


class ModelColumn(BaseModel):
    id: str
    name: str = ""
    description: str = ""
    type: str = ""
    # Typed rather than `list[str]`: an integer column's samples round-trip as
    # numbers, so a re-import can still tell 1 from '1' (see
    # ``auto_ontology.utils.sample_values.dump_sample_values``).
    sample_values: list[Any] = Field(default_factory=list)
    is_nullable: bool = True
    is_unique: bool = False


class ModelTable(BaseModel):
    id: str
    name: str = ""
    description: str = ""
    pk: list[str] = Field(default_factory=list)
    type: str = ""
    columns: list[ModelColumn] = Field(default_factory=list)


class ModelSchema(BaseModel):
    id: str
    name: str = ""
    database_name: str = ""
    tables: list[ModelTable] = Field(default_factory=list)


class ModelDatabase(BaseModel):
    id: str
    dialect: str = ""
    schemas: list[ModelSchema] = Field(default_factory=list)


class ModelForeignKey(BaseModel):
    source_column_id: str
    target_column_id: str


class ModelJoin(BaseModel):
    source_table_id: str
    target_table_id: str
    join_columns: list[dict[str, str]] = Field(default_factory=list)


class ModelDataLayer(BaseModel):
    databases: list[ModelDatabase] = Field(default_factory=list)
    foreign_keys: list[ModelForeignKey] = Field(default_factory=list)
    joins: list[ModelJoin] = Field(default_factory=list)


class ModelColumnAttribute(BaseModel):
    id: str
    name: str = ""
    description: str = ""
    column_id: str = ""


class ModelTerm(BaseModel):
    id: str
    name: str = ""
    description: str = ""
    represents: list[str] = Field(default_factory=list)
    columns_attributes: list[ModelColumnAttribute] = Field(default_factory=list)


class ModelSemanticFk(BaseModel):
    column_attribute_id: str
    column_id: str


class ModelSqlAttribute(BaseModel):
    id: str
    name: str = ""
    description: str = ""
    sql: str = ""
    sql_column_is: list[str] = Field(default_factory=list)
    term_id: str = ""


class ModelSqlAttributesBySource(BaseModel):
    manual: list[ModelSqlAttribute] = Field(default_factory=list)
    table: list[ModelSqlAttribute] = Field(default_factory=list)
    sql: list[ModelSqlAttribute] = Field(default_factory=list)
    bridge_table: list[ModelSqlAttribute] = Field(default_factory=list)


class ModelCustomAnalysis(BaseModel):
    id: str
    name: str = ""
    description: str = ""
    sql: str = ""
    sql_column_is: list[str] = Field(default_factory=list)


class ModelSemanticLayer(BaseModel):
    terms: list[ModelTerm] = Field(default_factory=list)
    semantic_fks: list[ModelSemanticFk] = Field(default_factory=list)
    sql_attributes: ModelSqlAttributesBySource = Field(
        default_factory=ModelSqlAttributesBySource,
    )
    custom_analyses: list[ModelCustomAnalysis] = Field(default_factory=list)


class AutoOntologyModelDocument(BaseModel):
    """Top-level Auto Ontology model document exchanged as YAML."""

    data_layer: ModelDataLayer = Field(default_factory=ModelDataLayer)
    semantic_layer: ModelSemanticLayer = Field(default_factory=ModelSemanticLayer)
    zones: list[dict[str, str]] = Field(default_factory=list)
