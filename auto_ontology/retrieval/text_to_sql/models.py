# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import List, Annotated, Literal


# ==================== TYPE ALIASES ====================

NonEmptyStr = Annotated[str, Field(min_length=1, description="Non-empty string")]

NonEmptyStrList = Annotated[
    list[str], Field(min_length=1, description="Non-empty list of strings")
]


class StrictModel(BaseModel):
    """Base model with strict validation settings."""

    model_config = ConfigDict(
        extra="forbid",  # forbid extra fields
        validate_assignment=True,  # re-check on assignment
        str_min_length=1,  # all strings must be non-empty by default
    )


# ==================== SCORE MODELS ====================


class ItemScore(BaseModel):
    """Represents a custom analysis item with classification."""

    id: NonEmptyStr
    label: Literal["custom_analysis", "column", "query", "table"] = Field(
        ...,
        description="The label of the custom analysis item",
    )
    classification: bool = Field(
        ...,
        description=(
            "True/False usage classification (True if the custom analysis was used in constructing "
            "the answer - either in SQL code or in deriving the answer from file contents/graph information)"
        ),
    )


NonEmptyItemScoreList = Annotated[
    List[ItemScore],
    Field(
        min_length=1, description="Non-empty list of custom analysis classifications"
    ),
]


class TableRemovalModel(BaseModel):
    """One table the relevance filter proposes to remove, with its checks.

    A table earns removal only by failing *both* tests, so both are answered
    per table rather than summarised once for the batch. An audit of 47
    harmful drops found 33 of them (70%) were tables the gold query used only
    to restrict rows, never to supply a projected column: asked for a single
    free-form justification the model reliably reasons about column
    provenance alone and stops there, which is exactly the blind spot.
    ``candidates_preparation`` enforces the two flags rather than trusting
    the prose, so a removal the model cannot justify is not applied.
    """

    model_config = ConfigDict(extra="forbid")

    table: str = Field(..., description="Name of the table to remove.")
    supplies_no_needed_column: bool = Field(
        ...,
        description=(
            "True only if this table supplies no column the answer needs — "
            "neither in the output, nor in a filter, nor in a grouping or sort."
        ),
    )
    cannot_change_qualifying_rows: bool = Field(
        ...,
        description=(
            "True only if removing this table cannot change WHICH rows qualify. "
            "Joining a table also restricts the result to rows that have a "
            "match in it, so this is False whenever the question limits results "
            "to rows having a related record there (only schools that appear in "
            "frpm; only patients who have an examination record) — even though "
            "none of that table's columns are returned."
        ),
    )
    justification: str = Field(
        ...,
        description=(
            "One sentence stating how you verified the two flags above, "
            "naming the row-scoping check explicitly."
        ),
    )


class TableRelevanceModel(BaseModel):
    """LLM output for table relevance filtering."""

    model_config = ConfigDict(extra="forbid")

    reasoning: str = Field(
        ...,
        description="Brief reasoning (1-2 sentences max) on which tables are relevant.",
    )
    tables_to_remove: list[TableRemovalModel] = Field(
        default_factory=list,
        description=(
            "Tables that can be safely removed, each with its removal checks. "
            "Leave EMPTY [] if unsure or none should be removed."
        ),
    )


class CustomAnalysisRelevanceModel(BaseModel):
    """LLM output for custom analysis relevance filtering."""

    model_config = ConfigDict(extra="forbid")

    reasoning: str = Field(
        ...,
        description="Brief reasoning (1-2 sentences max) on which custom analyses are relevant.",
    )
    analyses_to_remove: list[str] = Field(
        ...,
        description="Names of custom analyses that can be safely removed. Leave empty if unsure.",
    )


class ColumnAttributeSpec(BaseModel):
    """Typed result for a ColumnAttribute hit from the semantic VDB."""

    source_column: str
    name: str
    display_name: str = ""
    datatype: str = ""
    description: str | None = None


class CandidateFilterModel(BaseModel):
    """LLM structured output for intent-based candidate filtering."""

    model_config = ConfigDict(extra="forbid")

    reasoning: str = Field(..., description="Brief reasoning (1-2 sentences max).")
    best_id: str | None = Field(
        ...,
        description="ID of the single best candidate that matches the user's intent. Null if none match.",
    )


class CustomAnalysisFilterModel(BaseModel):
    """LLM structured output for filtering custom analysis candidates by relevance."""

    model_config = ConfigDict(extra="forbid")

    kept_ids: list[str] = Field(
        ...,
        description="IDs of custom analyses that are relevant to the user's question. Empty list if none are relevant.",
    )


class CombinedCandidateFilterModel(BaseModel):
    """LLM structured output for filtering both custom analyses and SQL attributes in one call."""

    model_config = ConfigDict(extra="forbid")

    custom_analysis_ids: list[str] = Field(
        ...,
        description="IDs of custom analyses to KEEP. Empty list if none are relevant.",
    )
    sql_attribute_ids: list[str] = Field(
        ...,
        description="IDs of SQL attributes to KEEP. Empty list if none are relevant.",
    )


class AnchorColumnModel(BaseModel):
    """LLM output for identifying the primary ColumnAttribute for the question."""

    model_config = ConfigDict(extra="forbid")

    reasoning: str = Field(..., description="Brief reasoning (1-2 sentences max).")
    anchor_id: str | None = Field(
        ...,
        description="The id of the ColumnAttribute that is the primary focus of the question.",
    )


class FollowUpResolutionModel(StrictModel):
    """Contextualize the current user question against completed prior turns."""

    is_follow_up: bool = Field(
        ...,
        description="Whether the current question depends on prior conversation context.",
    )
    standalone_question: NonEmptyStr = Field(
        ...,
        description=(
            "A complete standalone question. It must equal the current question "
            "when the question is independent."
        ),
    )


class SQLGenerationModel(StrictModel):
    """Model for SQL generation without formatting requirements.

    This model is used by SQL generation agents to return structured SQL data.
    Formatting is handled separately by SQLResponseFormattingAgent.

    Field order matters: the LLM fills fields sequentially, so ``thought``
    comes first to drain reasoning before it writes the clean output fields.
    """

    thought: str = Field(
        ...,
        description=(
            "Briefly explain the SQL approach and explicitly state every assumption "
            "made where the user's request or available schema does not uniquely "
            "determine the query. Do not omit assumptions. For each of the following "
            'that applies, state the choice AND the reason ("X, because Y"): '
            "(1) time window — the concrete boundary for any vague/relative time "
            "phrase (e.g. 'recently'); (2) zero/missing values — whether zero-count "
            "or NULL groups are included, excluded, or coerced to 0, and how any "
            "division guards against a zero denominator; (3) ties — what breaks a "
            "tie in a ranking/superlative query. If none were needed, state that "
            "explicitly."
        ),
    )
    sql_code: NonEmptyStr = Field(
        ...,
        description=(
            "The complete, executable SQL query. No comments, no delimiters, no explanation."
        ),
    )
    response: NonEmptyStr = Field(
        ...,
        description=(
            "User-facing summary in plain English (2-4 sentences): describe what "
            "is being calculated, which tables and columns are used, any filters "
            "or time windows applied, and the grouping/ordering. Refer to tables "
            "and columns by their human-readable names. "
            "Do NOT include SQL and code fences "
            "identifiers like 'schema.table', reasoning, self-corrections, "
            "formatting notes, or internal thoughts."
        ),
    )

    @field_validator("sql_code", "response")
    @classmethod
    def reject_placeholder_strings(cls, v: str, info) -> str:
        """Block LLM stubs like literal '...' that satisfy min length but are not valid output."""
        t = (v or "").strip()
        if t in ("...", "…", "..", ".") or (
            len(t) <= 3 and not t.isalnum() and set(t) <= {".", "…", " "}
        ):
            raise ValueError(
                f"{info.field_name!r} must be real content, not an ellipsis placeholder. "
                "sql_code must be the full executable statement; response must be a real explanation."
            )
        return v
