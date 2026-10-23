# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reasoning-based resolution of ambiguous entity-to-column candidates."""

from __future__ import annotations

from collections import defaultdict
import logging
from typing import Any, Dict

from langchain_core.messages import SystemMessage

from auto_ontology.dal.attributes import (
    fetch_attr_column_contexts,
    fetch_column_attribute_columns_map,
    fetch_semantic_fk_join_groups,
)
from auto_ontology.dal.terms import fetch_term_synonyms, fetch_term_table_pairs
from auto_ontology.retrieval.text_to_sql.base import BaseAgent
from auto_ontology.retrieval.text_to_sql.models import (
    ColumnAmbiguityResolutionModel,
)
from auto_ontology.retrieval.text_to_sql.state import (
    AgentState,
    get_question_for_processing,
)
from auto_ontology.utils.llm_invoke import invoke_with_structured_output

logger = logging.getLogger(__name__)

_MAX_TEXT_CHARS = 700
_MAX_RELATIONSHIPS_PER_ATTRIBUTE = 6

_RESOLUTION_PROMPT = """\
You resolve ambiguous semantic ColumnAttribute candidates for field and exact
value entities.
Reason carefully as a human interpreting the request and select at most one
candidate in every group.

Selection priority:
1. EVIDENCE IS PRIMARY. When the evidence explicitly identifies or defines the
   field for a group, follow it.
2. Otherwise infer the intended field from the original question. Determine
   which subject the phrase modifies and the exact role or meaning expressed by
   the wording and its local grammatical context.
3. Evaluate that interpretation against the semantic layer: attribute and term
   meaning, physical ownership, subject relationship, synonyms, and semantic-FK
   roles. Reject candidates whose table role or relationship conflicts with
   the intended subject.
4. Use vector distance only as a final tie-breaker.

Do not default automatically to either a generic or specialized column. Do not
transfer context from an unrelated condition to the field being resolved. Avoid
a related-table attribute when the question does not ask for that relationship.
If the evidence, question, and semantic layer do not support one candidate,
return null for that group rather than guessing.

For a field entity, the selection identifies the column required by the
question. For an exact-value entity, the selection identifies the column in
which that value is intended to be used; database probing will verify that the
value actually exists there.

Original question:
{original_question}

Normalized question:
{normalized_question}

Evidence:
{evidence}

Extracted primary subject:
{subject}

Ambiguous entity groups:
{groups}
"""


def _compact(value: object, limit: int = _MAX_TEXT_CHARS) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else f"{text[: limit - 1]}…"


def _spec_key(spec: dict[str, Any]) -> tuple[str, str, str] | None:
    entity_type = str(spec.get("type") or "").strip()
    original_text = str(spec.get("original_text") or "").strip()
    meaning = str(spec.get("meaning") or "").strip()
    if entity_type not in {"field", "exact_value"} or not original_text or not meaning:
        return None
    return entity_type, original_text.casefold(), meaning.casefold()


def _entity_specs(hit: dict) -> list[dict[str, str]]:
    raw_specs = hit.get("query_entity_specs")
    if not isinstance(raw_specs, list):
        raw_spec = hit.get("query_entity_spec")
        raw_specs = [raw_spec] if isinstance(raw_spec, dict) else []

    specs: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for raw_spec in raw_specs:
        if not isinstance(raw_spec, dict):
            continue
        key = _spec_key(raw_spec)
        if key is None or key in seen:
            continue
        seen.add(key)
        specs.append(
            {
                "type": str(raw_spec["type"]).strip(),
                "original_text": str(raw_spec["original_text"]).strip(),
                "meaning": str(raw_spec["meaning"]).strip(),
            }
        )

    if specs:
        return specs

    query_entity = str(hit.get("query_entity") or "").strip()
    query_type = str(hit.get("query_entity_type") or "field")
    query_meaning = str(hit.get("query_entity_meaning") or query_entity).strip()
    if query_type in {"field", "exact_value"} and query_entity and query_meaning:
        return [
            {
                "type": query_type,
                "original_text": query_entity,
                "meaning": query_meaning,
            }
        ]
    return []


def _has_value_or_evidence_provenance(hit: dict) -> bool:
    """Whether narrowing a field must not remove this hit globally."""
    if hit.get("query_evidence"):
        return True
    raw_specs = hit.get("query_entity_specs")
    if not isinstance(raw_specs, list):
        raw_spec = hit.get("query_entity_spec")
        raw_specs = [raw_spec] if isinstance(raw_spec, dict) else []
    if any(
        isinstance(spec, dict) and spec.get("type") == "exact_value"
        for spec in raw_specs
    ):
        return True
    return hit.get("query_entity_type") == "exact_value"


def _ambiguous_groups(
    hits: list[dict],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for hit in hits:
        for spec in _entity_specs(hit):
            key = _spec_key(spec)
            if key is None:
                continue
            entry = grouped.setdefault(key, {"spec": spec, "hits": {}})
            hit_id = str(hit.get("id") or "")
            if hit_id:
                entry["hits"].setdefault(hit_id, hit)
    return {key: entry for key, entry in grouped.items() if len(entry["hits"]) > 1}


def _safe_semantic_context(
    attr_ids: list[str],
    *,
    database_name: str | None,
    subject_term_id: str | None,
) -> tuple[
    dict[str, dict],
    dict[str, list[str]],
    dict[str, dict],
    list[dict],
    dict[str, set[str]],
]:
    try:
        contexts = fetch_attr_column_contexts(
            attr_ids,
            database_name=database_name,
        )
    except Exception:
        logger.warning("Failed to fetch attribute contexts", exc_info=True)
        contexts = {}
    try:
        synonyms = fetch_term_synonyms(attr_ids)
    except Exception:
        logger.warning("Failed to fetch term synonyms", exc_info=True)
        synonyms = {}
    try:
        columns_map = fetch_column_attribute_columns_map(attr_ids)
    except Exception:
        logger.warning("Failed to fetch semantic column relationships", exc_info=True)
        columns_map = {}
    try:
        semantic_fks = (
            fetch_semantic_fk_join_groups(
                database_name,
                attribute_ids=attr_ids,
            )
            if database_name
            else []
        )
    except Exception:
        logger.warning("Failed to fetch semantic FK context", exc_info=True)
        semantic_fks = []

    subject_paths: dict[str, set[str]] = defaultdict(set)
    if subject_term_id:
        try:
            for pair in fetch_term_table_pairs(
                database_name=database_name,
                term_ids=[subject_term_id],
            ):
                table_id = str(pair.get("table_id") or "")
                path = str(pair.get("path") or "")
                if table_id and path:
                    subject_paths[table_id].add(path)
        except Exception:
            logger.warning("Failed to fetch subject table context", exc_info=True)

    return contexts, synonyms, columns_map, semantic_fks, dict(subject_paths)


def _format_column_ref(column: dict | None) -> str:
    if not column:
        return "(none)"
    table_name = str(column.get("table_name") or "")
    column_name = str(column.get("column_name") or "")
    return ".".join(part for part in (table_name, column_name) if part) or "(unknown)"


def _format_candidate(
    hit: dict,
    contexts: dict[str, dict],
    synonyms: dict[str, list[str]],
    columns_map: dict[str, dict],
    semantic_fks_by_attr: dict[str, list[dict]],
    subject_paths: dict[str, set[str]],
) -> str:
    attr_id = str(hit.get("id") or "")
    context = contexts.get(attr_id, {})
    physical = ".".join(
        str(context.get(key) or "")
        for key in ("schema_name", "table_name", "col_name")
        if context.get(key)
    )
    term_name = str(context.get("term_name") or "")
    term_synonyms = ", ".join(synonyms.get(term_name, [])) or "(none)"
    table_id = str(context.get("table_id") or "")
    subject_links = ", ".join(sorted(subject_paths.get(table_id, set()))) or "(none)"
    relationship = columns_map.get(attr_id, {})
    referenced = [
        _format_column_ref(column)
        for column in relationship.get("referenced_columns", [])[
            :_MAX_RELATIONSHIPS_PER_ATTRIBUTE
        ]
    ]
    fk_lines = []
    for row in semantic_fks_by_attr.get(attr_id, [])[:_MAX_RELATIONSHIPS_PER_ATTRIBUTE]:
        fk_lines.append(
            f"{row.get('source_table_name')}.{row.get('source_column_name')} -> "
            f"{row.get('target_table_name')}.{row.get('target_column_name')} "
            f"(default_for_joins={bool(row.get('use_in_join_paths'))}; "
            f"source_meaning={_compact(row.get('source_column_description'), 180) or '(none)'}; "
            f"target_meaning={_compact(row.get('target_column_description'), 180) or '(none)'})"
        )

    return "\n".join(
        [
            f"  Candidate id: {attr_id}",
            f"    vector_distance: {hit.get('score')}",
            f"    semantic_match: {_compact(hit.get('text')) or '(none)'}",
            f"    attribute: {context.get('attr_name') or hit.get('name') or '(unknown)'}",
            f"    attribute_description: {_compact(context.get('attr_description')) or '(none)'}",
            f"    term: {term_name or '(none)'}",
            f"    term_description: {_compact(context.get('term_description')) or '(none)'}",
            f"    term_synonyms: {term_synonyms}",
            f"    physical_column: {physical or '(unknown)'}",
            f"    table_description: {_compact(context.get('table_description')) or '(none)'}",
            f"    column_description: {_compact(context.get('col_description')) or '(none)'}",
            f"    datatype: {context.get('datatype') or hit.get('data_type') or '(unknown)'}",
            f"    primary_key: {bool(context.get('is_primary_key'))}",
            f"    subject_semantic_links: {subject_links}",
            f"    defining_column: {_format_column_ref(relationship.get('primary_column'))}",
            f"    referenced_by: {', '.join(referenced) if referenced else '(none)'}",
            f"    semantic_fk_roles: {'; '.join(fk_lines) if fk_lines else '(none)'}",
            f"    retrieved_from_evidence: {bool(hit.get('query_evidence'))}",
        ]
    )


def _format_groups(
    groups: dict[tuple[str, str, str], dict[str, Any]],
    contexts: dict[str, dict],
    synonyms: dict[str, list[str]],
    columns_map: dict[str, dict],
    semantic_fks: list[dict],
    subject_paths: dict[str, set[str]],
) -> tuple[str, dict[str, tuple[str, str, str]]]:
    semantic_fks_by_attr: dict[str, list[dict]] = defaultdict(list)
    for row in semantic_fks:
        attr_id = str(row.get("target_attribute_id") or "")
        if attr_id:
            semantic_fks_by_attr[attr_id].append(row)

    blocks: list[str] = []
    keys_by_group_id: dict[str, tuple[str, str, str]] = {}
    for index, (key, entry) in enumerate(groups.items(), start=1):
        group_id = f"group_{index}"
        keys_by_group_id[group_id] = key
        spec = entry["spec"]
        candidate_block = "\n".join(
            _format_candidate(
                hit,
                contexts,
                synonyms,
                columns_map,
                semantic_fks_by_attr,
                subject_paths,
            )
            for hit in entry["hits"].values()
        )
        blocks.append(
            f"{group_id}\n"
            f"  entity_type: {spec['type']}\n"
            f"  entity_text: {spec['original_text']}\n"
            f"  entity_meaning: {spec['meaning']}\n"
            f"{candidate_block}"
        )
    return "\n\n".join(blocks), keys_by_group_id


def _apply_selections(
    hits: list[dict],
    groups: dict[tuple[str, str, str], dict[str, Any]],
    selected_by_key: dict[tuple[str, str, str], str],
) -> list[dict]:
    if not selected_by_key:
        return hits

    kept: list[tuple[int, dict]] = []
    preferred_value_ids = {
        selected_id
        for key, selected_id in selected_by_key.items()
        if key[0] == "exact_value"
    }
    for index, hit in enumerate(hits):
        if _has_value_or_evidence_provenance(hit):
            kept.append((index, hit))
            continue
        field_keys = {
            key
            for spec in _entity_specs(hit)
            if (key := _spec_key(spec)) is not None
            and key[0] == "field"
            and key in groups
        }
        if not field_keys:
            kept.append((index, hit))
            continue
        hit_id = str(hit.get("id") or "")
        if any(
            key not in selected_by_key or selected_by_key[key] == hit_id
            for key in field_keys
        ):
            kept.append((index, hit))

    kept.sort(
        key=lambda indexed_hit: (
            0 if str(indexed_hit[1].get("id") or "") in preferred_value_ids else 1,
            indexed_hit[0],
        )
    )
    return [hit for _, hit in kept]


class ColumnAmbiguityResolverAgent(BaseAgent):
    """Resolve ambiguous entity candidates using evidence and semantic context."""

    def __init__(self) -> None:
        super().__init__("column_ambiguity_resolver")

    def execute(self, state: AgentState) -> Dict[str, Any]:
        path_state = state.get("path_state", {})
        hits = list(path_state.get("retrieved_column_attributes") or [])
        groups = _ambiguous_groups(hits)
        if not groups:
            return {"path_state": path_state}

        attr_ids = list(
            dict.fromkeys(
                attr_id for entry in groups.values() for attr_id in entry["hits"]
            )
        )
        subject_term = path_state.get("retrieved_subject_term") or {}
        subject_term_id = str(subject_term.get("id") or "") or None
        target_db = str(path_state.get("target_db") or "") or None
        (
            contexts,
            synonyms,
            columns_map,
            semantic_fks,
            subject_paths,
        ) = _safe_semantic_context(
            attr_ids,
            database_name=target_db,
            subject_term_id=subject_term_id,
        )
        groups_block, keys_by_group_id = _format_groups(
            groups,
            contexts,
            synonyms,
            columns_map,
            semantic_fks,
            subject_paths,
        )

        original_question = str(state.get("initial_question") or "").strip()
        normalized_question = get_question_for_processing(state)
        evidence = str(state.get("evidence") or "").strip()
        subject = str(path_state.get("subject") or "").strip()
        messages = [
            SystemMessage(
                content=_RESOLUTION_PROMPT.format(
                    original_question=original_question or normalized_question,
                    normalized_question=normalized_question,
                    evidence=evidence or "(none)",
                    subject=subject or "(not identified)",
                    groups=groups_block,
                )
            )
        ]

        try:
            result = invoke_with_structured_output(
                state["llm"],
                messages,
                ColumnAmbiguityResolutionModel,
            )
        except Exception:
            self.logger.warning(
                "Column ambiguity resolution failed; retaining all candidates",
                exc_info=True,
            )
            return {"path_state": path_state}
        if result is None:
            return {"path_state": path_state}

        selected_by_key: dict[tuple[str, str, str], str] = {}
        seen_group_ids: set[str] = set()
        for selection in result.selections:
            group_id = selection.group_id
            if group_id in seen_group_ids:
                continue
            seen_group_ids.add(group_id)
            key = keys_by_group_id.get(group_id)
            selected_id = selection.selected_attribute_id
            if key is None or selected_id is None:
                continue
            valid_ids = set(groups[key]["hits"])
            if selected_id not in valid_ids:
                continue
            selected_by_key[key] = selected_id
            self.logger.info(
                "Resolved %s entity %r to attribute %s: %s",
                key[0],
                groups[key]["spec"]["original_text"],
                selected_id,
                selection.reasoning,
            )

        path_state["retrieved_column_attributes"] = _apply_selections(
            hits,
            groups,
            selected_by_key,
        )
        return {"path_state": path_state}


__all__ = ["ColumnAmbiguityResolverAgent"]
