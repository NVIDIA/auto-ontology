"""Pydantic DTOs for LLM structured I/O — not an in-memory graph store."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


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
    """LLM output: columns that may be FKs but lack graph FOREIGN_KEY edges."""

    model_config = ConfigDict(extra="forbid")

    suggestions: list[PotentialFkSuggestion] = Field(
        default_factory=list,
        description="Suspected FK columns; empty when none apply.",
    )


class FkAndPkResult(BaseModel):
    """LLM output: FK suggestions plus columns that look like the table's own PK."""

    model_config = ConfigDict(extra="forbid")

    fk_suggestions: list[PotentialFkSuggestion] = Field(
        default_factory=list,
        description="Suspected FK columns; empty when none apply.",
    )
    pk_column_names: list[str] = Field(
        default_factory=list,
        description=(
            "Column names that appear to be the table's own primary key "
            "even if not declared as such. Usually empty or one entry."
        ),
    )


class FkHitSelection(BaseModel):
    """LLM output: selects the best matching Column hit from a VDB result list."""

    model_config = ConfigDict(extra="forbid")

    selected_id: str | None = Field(
        ...,
        description=(
            "The neo4j_id of the candidate that is the primary-key column this FK "
            "references, exactly as shown in the candidate list. "
            "Return null if none of the candidates are a confident match."
        ),
    )
    rationale: str = Field(default="")


# ---------------------------------------------------------------------------
# Auto SqlAttribute extraction
# ---------------------------------------------------------------------------


class SqlAttributeProposal(BaseModel):
    """LLM output: one derived business metric built from table columns."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        ...,
        description=(
            "A realistic natural-language question a business user would ask "
            'that this metric answers (e.g. "What is the profit margin?").'
        ),
    )
    name: str = Field(
        ...,
        description=(
            "User-friendly metric name with spaces between words "
            "(e.g. Net Revenue, Profit Margin)."
        ),
    )
    description: str = Field(
        ...,
        description="Short business definition of this metric.",
    )
    expression: str = Field(
        ...,
        description=(
            "A valid SQL SELECT statement that computes the metric using "
            "qualified column references (schema.table.column). "
            "Must reference columns from the provided table only."
        ),
    )


class SqlAttributeExtractionResult(BaseModel):
    """LLM output: derived metrics for a single table."""

    model_config = ConfigDict(extra="forbid")

    metrics: list[SqlAttributeProposal] = Field(
        default_factory=list,
        description=(
            "Non-trivial derived business metrics that combine two or more columns. "
            "Return an empty list when no meaningful metrics can be derived. "
            "Only include metrics that are genuinely useful for business analysis."
        ),
    )
