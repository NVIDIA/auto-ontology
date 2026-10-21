# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Add the compiled identifier for the question's subject as evidence."""

from __future__ import annotations

from typing import Any, Dict

from auto_ontology.retrieval.text_to_sql.base import BaseAgent
from auto_ontology.retrieval.text_to_sql.connector_routing import (
    resolve_connector_from_tables,
)
from auto_ontology.retrieval.text_to_sql.state import AgentState
from auto_ontology.utils.sql_identifiers import qualified_name, quoted_identifier

_EVIDENCE_PREFIX = "Subject identifier rule:"


def _subject_table(path_state: dict[str, Any]) -> dict[str, Any] | None:
    subject_ids = {
        str(table_id)
        for table_id in path_state.get("subject_table_ids") or []
        if table_id
    }
    if not subject_ids:
        return None

    candidates = [
        table
        for table in path_state.get("relevant_tables") or []
        if str(table.get("id") or "") in subject_ids and table.get("default_identifier")
    ]
    if len(candidates) == 1:
        return candidates[0]

    anchor_table_id = str(
        (path_state.get("primary_attribute") or {}).get("table_id") or ""
    )
    anchored = [
        table for table in candidates if str(table.get("id") or "") == anchor_table_id
    ]
    return anchored[0] if len(anchored) == 1 else None


def _identifier_rule(
    table: dict[str, Any],
    *,
    subject: str,
    target_db: str | None,
    connector: Any | None,
) -> str:
    schema = str(table.get("schema_name") or "")
    table_name = str(table.get("name") or "")
    database = str(table.get("database_name") or target_db or "")
    dialect = getattr(connector, "dialect", None)
    qualified_table = (
        connector.qualify(schema or None, table_name)
        if connector is not None
        else qualified_name(database, schema, table_name, dialect=dialect)
    )
    identifier = (
        f"{qualified_table}."
        f"{quoted_identifier(str(table['default_identifier']), dialect)}"
    )
    subject_label = subject or table_name
    return (
        f'{_EVIDENCE_PREFIX} When identifying the subject "{subject_label}" without '
        f"an explicitly requested output field, return only {identifier}. Explicitly "
        "requested fields, aggregates, and requested properties take precedence."
    )


class IdentifierEvidenceAgent(BaseAgent):
    """Append one deterministic identifier rule for the resolved subject table."""

    def __init__(self) -> None:
        super().__init__("identifier_evidence")

    def validate_input(self, state: AgentState) -> bool:
        return True

    def execute(self, state: AgentState) -> Dict[str, Any]:
        if not state.get("shorten_answer", False):
            return {}

        path_state = dict(state.get("path_state") or {})
        table = _subject_table(path_state)
        if table is None:
            self.logger.info(
                "No unambiguous subject-table identifier available for evidence"
            )
            return {}

        tables = list(path_state.get("relevant_tables") or [])
        connector = resolve_connector_from_tables(tables, state.get("connectors") or [])
        rule = _identifier_rule(
            table,
            subject=str(path_state.get("subject") or "").strip(),
            target_db=path_state.get("target_db"),
            connector=connector,
        )
        evidence = str(state.get("evidence") or "").rstrip()
        if rule in evidence:
            return {}
        combined = f"{evidence}\n{rule}" if evidence else rule
        path_state["subject_identifier_evidence"] = rule
        self.logger.info(
            "Added default identifier evidence for subject table %r", table.get("name")
        )
        return {"evidence": combined, "path_state": path_state}


__all__ = ["IdentifierEvidenceAgent"]
