# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic DTOs for LLM structured I/O — not an in-memory graph store."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


@dataclass
class ProcessTableResult:
    """Names of nodes created/merged for a single table."""

    term_names: list[str] = field(default_factory=list)
    attr_names: list[str] = field(default_factory=list)


class TermColumnRef(BaseModel):
    """LLM output: column assignment with a user-friendly display label."""

    model_config = ConfigDict(extra="forbid")

    source_column: str = Field(
        ...,
        description="Physical column name — must match a candidate column.",
    )
    display_name: str = Field(
        ...,
        description=(
            "User-friendly ColumnAttribute label with spaces between words "
            "(e.g. Order Date, Total Amount)."
        ),
    )


class TermAttributeAssignment(BaseModel):
    """Resolved column assignment on a Term."""

    source_column: str
    display_name: str


class RawTermProposal(BaseModel):
    """LLM output: one business Term with column assignments only."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        description=(
            "User-friendly business Term name with spaces between words "
            "(e.g. Purchase Order)."
        ),
    )
    description: str = Field(default="", description="Short business definition.")
    attributes: list[TermColumnRef] = Field(
        default_factory=list,
        description=(
            "Candidate columns assigned to this Term with user-friendly display names."
        ),
    )


class RawTableTermsResult(BaseModel):
    """LLM output: Terms and column assignments for a single physical table."""

    model_config = ConfigDict(extra="forbid")

    terms: list[RawTermProposal] = Field(
        ...,
        description=(
            "Usually one Term. Propose multiple only when columns clearly belong "
            "to distinct business concepts."
        ),
    )


class TermProposal(BaseModel):
    """Sanitized Term with resolved ColumnAttribute assignments."""

    name: str
    description: str = ""
    synonyms: list[str] = Field(default_factory=list)
    attributes: list[TermAttributeAssignment] = Field(default_factory=list)


class TableTermsResult(BaseModel):
    """Sanitized Terms and column assignments for a single physical table."""

    terms: list[TermProposal] = Field(...)


class SeedSelectionResult(BaseModel):
    """LLM output: single seed table for BFS."""

    model_config = ConfigDict(extra="forbid")

    table_name: str = Field(..., description="Physical table name.")
    rationale: str = Field(default="")


class ColumnAttributeSpec(BaseModel):
    """Column candidate for Term assignment; display_name set by extract_term."""

    source_column: str
    name: str
    display_name: str = ""
    datatype: str = ""
    description: str | None = None


class ColumnDescription(BaseModel):
    """LLM-generated business description for a single column."""

    model_config = ConfigDict(extra="forbid")

    column_name: str = Field(
        ...,
        description="Physical column name — must match a provided column.",
    )
    description: str = Field(
        default="",
        description="One concise sentence describing what the column represents.",
    )


class ColumnDescriptionResult(BaseModel):
    """LLM output: business descriptions for a table's columns."""

    model_config = ConfigDict(extra="forbid")

    descriptions: list[ColumnDescription] = Field(
        default_factory=list,
        description="One entry per column provided in the prompt.",
    )


class PotentialFkSuggestion(BaseModel):
    """One column the LLM suspects is a foreign key."""

    model_config = ConfigDict(extra="forbid")

    column_name: str = Field(
        ...,
        description="Physical column name that likely references another table.",
    )
    rationale: str = Field(
        default="",
        description="Brief reason this column looks like a foreign key.",
    )


class PotentialFkResult(BaseModel):
    """Sanitized FK suggestions and table-grain classification."""

    model_config = ConfigDict(extra="forbid")

    suggestions: list[PotentialFkSuggestion] = Field(
        default_factory=list,
        description="Suspected FK columns; empty when none apply.",
    )
    is_junction_table: bool = Field(
        default=False,
        description="Whether each row represents an association between entity roles.",
    )
    junction_table_rationale: str = Field(
        default="",
        description="Brief reason for the junction-table decision.",
    )


class FkAndPkResult(BaseModel):
    """LLM output: FK/PK suggestions plus table-grain classification."""

    model_config = ConfigDict(extra="forbid")

    fk_suggestions: list[PotentialFkSuggestion] = Field(
        default_factory=list,
        description="Suspected FK columns; empty when none apply.",
    )
    pk_column_names: list[str] = Field(
        default_factory=list,
        description=(
            "Columns that identify this table's own records: a bare id or uuid, "
            "or a column named exactly <this_table>_id, <this_table>_key, or "
            "<this_table>_code. A *_key, *_id, or *_code column can also be this "
            "table's own key when it identifies this table's rows; the primary "
            "key itself uses that suffix. Never another table's own key, and "
            "never a column marked is_unique false. Usually empty or one entry."
        ),
    )
    is_junction_table: bool = Field(
        default=False,
        description=(
            "True only when each row represents an association between at least "
            "two entity roles."
        ),
    )
    junction_table_rationale: str = Field(
        default="",
        description="Brief reason for the junction-table decision.",
    )


class FkHitSelection(BaseModel):
    """LLM output: selects the best matching Column hit from a VDB result list."""

    model_config = ConfigDict(extra="forbid")

    selected_id: str | None = Field(
        ...,
        description=(
            "The column_id of the candidate that is the primary-key column this FK "
            "references, exactly as shown in the candidate list. "
            "Return null if none of the candidates are a confident match."
        ),
    )
    rationale: str = Field(default="")


class LegacyTableJudgement(BaseModel):
    """LLM output: which of two near-identical tables is the outdated copy."""

    model_config = ConfigDict(extra="forbid")

    legacy_table_id: str | None = Field(
        ...,
        description=(
            "The exact table_id of the outdated table, as shown in the prompt, "
            "or null when the evidence does not show either one is outdated."
        ),
    )
    reason: str = Field(default="", description="Brief reason for the decision.")


class TableRoleJudgement(BaseModel):
    """LLM output: a table's role in the schema and what one row represents."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["fact", "dimension", "junction"] | None = Field(
        ...,
        description=(
            "fact: rows record events, transactions, offerings or measurements, "
            "usually with several references and numeric measures. dimension: "
            "one row per entity or code that other tables reference for its "
            "name and attributes. junction: each row only associates two or "
            "more entities. Null when the evidence does not say."
        ),
    )
    grain: str = Field(
        default="",
        description=(
            "One short sentence naming what a single row represents, for "
            "example 'one row per subject offered in a term'."
        ),
    )


class SiblingTableNotes(BaseModel):
    """LLM output: how two near-identical tables differ."""

    model_config = ConfigDict(extra="forbid")

    first_note: str | None = Field(
        ...,
        description=(
            "When to use the first table rather than the second, in one short "
            "sentence. Null when the two tables cannot be confused."
        ),
    )
    second_note: str | None = Field(
        ...,
        description=(
            "When to use the second table rather than the first, in one short "
            "sentence. Null when the two tables cannot be confused."
        ),
    )


class SemanticFkJoinDefaultSelection(BaseModel):
    """LLM output selecting one FK column as the default join relationship."""

    model_config = ConfigDict(extra="forbid")

    selected_column_id: str | None = Field(
        ...,
        description=(
            "The exact source-column ID of the relationship that best represents "
            "the default join between the supplied tables, or null when uncertain."
        ),
    )
    rationale: str = Field(default="")
