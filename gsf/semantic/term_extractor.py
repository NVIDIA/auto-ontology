"""LLM Term extraction — names and column display-name assignments only."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from gsf.semantic.deterministic import to_term_name
from gsf.semantic.domain import DomainSummary
from gsf.utils.llm_invoke import (
    get_non_reasoning_llm_client,
    invoke_with_structured_output,
)
from pydantic import BaseModel, ConfigDict, Field

from gsf.semantic.models import (
    ColumnAttributeSpec,
    RawTableTermsResult,
    TableTermsResult,
    TermAttributeAssignment,
    TermProposal,
)

_INVALID = {"unnamed", "unknown", "none", ""}
_LABEL_RE = re.compile(r"^[A-Z][A-Za-z0-9]*(?: [A-Za-z][A-Za-z0-9]*)*$")

_SYSTEM = """\
You propose business Terms for a relational table and assign candidate columns \
to each Term with user-friendly display labels.

Rules:
1. Strongly prefer ONE Term per table. A second Term is warranted ONLY when the \
table is a true junction between two unrelated entities. Never split a single \
entity into separate Terms for its core fields versus its metadata, lifecycle, \
or audit fields — those all belong to the same Term. If in doubt, use one Term.
2. Term names must be user-friendly with spaces between words (e.g. Purchase Order, \
not purchase_orders or PurchaseOrder).
3. Assign EVERY candidate column to exactly one Term. For each assignment return \
source_column exactly as given and a display_name — a user-friendly ColumnAttribute \
label with spaces between words (e.g. Order Date, Total Amount).
4. Do not propose IS_A, PART_OF, or ROLE relationships.
5. If the table name contains a word that indicates it is a variant of another table \
(e.g. ARC, ARCHIVE, HIST, HISTORY, STAGING, DELETED, TEMP), prefix every display_name \
with a qualifying word derived from that suffix \
(e.g. PEOPLE_ARC → "Archived Is Sales Person", ORDERS_HIST → "Historical Total Amount")."""


class _SynonymExtractionModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    synonyms: list[str] = Field(
        default_factory=list,
        description=(
            "Abbreviations, acronyms, or alternate names explicitly written in the "
            "description. Empty list if none are stated."
        ),
    )


_SYNONYM_SYSTEM = """\
Your only task is to extract alternate names for a business entity from a table description.

Look for abbreviations, acronyms, or aliases that are explicitly written — for example \
through phrases like "also referred to as", "also known as", "abbreviated as", \
"short for", "stands for", or a parenthetical directly after the entity name such as \
"business unit (BU)".

Return each alternate name verbatim as a separate entry in synonyms.
Example: "business unit (BU)" → synonyms: ["BU"]

Return an empty list if no alternate names are explicitly stated. \
Do NOT infer, guess, or add any name that is not directly written in the description."""


def _normalize_label(name: str, *, fallback: str) -> str:
    cleaned = name.strip()
    if cleaned.lower() in _INVALID or not _LABEL_RE.match(cleaned):
        return fallback
    return to_term_name(cleaned)


def _format_spec_line(spec: ColumnAttributeSpec) -> str:
    desc = f" — {spec.description}" if spec.description else ""
    return f"  - source_column={spec.source_column} ({spec.datatype}){desc}"


def apply_display_names_to_specs(
    table_result: TableTermsResult,
    specs: list[ColumnAttributeSpec],
) -> None:
    """Write LLM display labels back onto column specs for downstream merges."""
    by_column = {spec.source_column: spec for spec in specs}
    for term in table_result.terms:
        for attr in term.attributes:
            spec = by_column.get(attr.source_column)
            if spec is not None:
                spec.display_name = attr.display_name
    for spec in specs:
        if not spec.display_name:
            spec.display_name = spec.name


def _fallback_result(
    table: dict[str, Any],
    specs: list[ColumnAttributeSpec],
) -> TableTermsResult:
    name = to_term_name(table["name"])
    attributes = [
        TermAttributeAssignment(
            source_column=spec.source_column,
            display_name=spec.name,
        )
        for spec in specs
    ]
    result = TableTermsResult(
        terms=[
            TermProposal(
                name=name,
                description=f"Business entity represented by table {table['name']}",
                attributes=attributes,
            )
        ]
    )
    apply_display_names_to_specs(result, specs)
    return result


def _sanitize_result(
    result: RawTableTermsResult,
    *,
    table: dict[str, Any],
    specs: list[ColumnAttributeSpec],
) -> TableTermsResult:
    spec_by_column = {spec.source_column: spec for spec in specs}
    allowed_columns = set(spec_by_column)
    default_term = to_term_name(table["name"])

    sanitized_terms: list[TermProposal] = []
    seen_term_names: set[str] = set()
    assigned_columns: set[str] = set()

    for raw_term in result.terms:
        term_name = _normalize_label(raw_term.name, fallback=default_term)
        if term_name in seen_term_names:
            continue
        seen_term_names.add(term_name)

        attributes: list[TermAttributeAssignment] = []
        for raw_attr in raw_term.attributes:
            source_column = raw_attr.source_column.strip()
            if (
                not source_column
                or source_column not in allowed_columns
                or source_column in assigned_columns
            ):
                continue
            spec = spec_by_column[source_column]
            attributes.append(
                TermAttributeAssignment(
                    source_column=source_column,
                    display_name=_normalize_label(
                        raw_attr.display_name,
                        fallback=spec.name,
                    ),
                )
            )
            assigned_columns.add(source_column)

        sanitized_terms.append(
            TermProposal(
                name=term_name,
                description=raw_term.description.strip(),
                attributes=attributes,
            )
        )

    if not sanitized_terms:
        return _fallback_result(table, specs)

    # Any unassigned columns fall back to the first Term.
    primary = sanitized_terms[0]
    for spec in specs:
        if spec.source_column not in assigned_columns:
            primary.attributes.append(
                TermAttributeAssignment(
                    source_column=spec.source_column,
                    display_name=spec.name,
                )
            )
            assigned_columns.add(spec.source_column)

    table_result = TableTermsResult(terms=sanitized_terms)
    apply_display_names_to_specs(table_result, specs)
    return table_result


def _extract_synonyms(description: str) -> list[str]:
    """Dedicated LLM call to extract alternate names from a table description."""
    if not description:
        return []
    result = invoke_with_structured_output(
        get_non_reasoning_llm_client(temperature=0.0, max_tokens=1024),
        [
            SystemMessage(content=_SYNONYM_SYSTEM),
            HumanMessage(content=f"Description: {description}"),
        ],
        _SynonymExtractionModel,
    )
    if result is None:
        return []
    return [s.strip() for s in result.synonyms if s.strip()]


def extract_term(
    table: dict[str, Any],
    ctx: dict[str, Any],
    specs: list[ColumnAttributeSpec],
    *,
    domain_summary: DomainSummary | None = None,
) -> TableTermsResult:
    """Propose one or more Terms and assign candidate column attributes to each."""
    if not specs:
        return _fallback_result(table, specs)

    description = table.get("description") or ""

    spec_lines = "\n".join(_format_spec_line(spec) for spec in specs[:40])
    domain_block = ""
    if domain_summary:
        domain_block = (
            f"\nDomains: {', '.join(domain_summary.domains[:8])}\n"
            f"Core entities: {', '.join(domain_summary.core_entities[:12])}\n"
        )
    prompt = (
        f"Table: {table['name']}\n"
        f"Description: {description}\n"
        f"Candidate columns (assign each to exactly one Term with a display_name):\n"
        f"{spec_lines}\n"
        f"{domain_block}"
    )
    result = invoke_with_structured_output(
        get_non_reasoning_llm_client(temperature=0.0, max_tokens=4096),
        [SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)],
        RawTableTermsResult,
    )
    if result is None:
        return _fallback_result(table, specs)

    table_result = _sanitize_result(result, table=table, specs=specs)

    synonyms = _extract_synonyms(description)
    if synonyms:
        for term in table_result.terms:
            term.synonyms = synonyms

    return table_result
