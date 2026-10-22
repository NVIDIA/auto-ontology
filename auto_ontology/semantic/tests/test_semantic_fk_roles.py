# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock

from auto_ontology.semantic import semantic_fk_roles
from auto_ontology.semantic.semantic_fk_role_tools import (
    SemanticFkRoleToolName,
    run_semantic_fk_role_tool,
)
from auto_ontology.semantic.semantic_fk_roles import (
    SemanticFkRoleAction,
    SemanticFkRoleActionType,
    SemanticFkRoleProposal,
    SemanticFkRoleToolArguments,
)


def _edge(
    source_table_id: str,
    target_table_id: str,
    *,
    source_column_id: str | None = None,
    target_attribute_id: str | None = None,
) -> dict[str, object]:
    source_column_id = source_column_id or f"{source_table_id}-column"
    target_attribute_id = target_attribute_id or f"{target_table_id}-attribute"
    return {
        "source_table_id": source_table_id,
        "source_table_name": source_table_id,
        "source_column_id": source_column_id,
        "source_column_name": "parent_id",
        "target_attribute_id": target_attribute_id,
        "target_attribute_name": "ID",
        "target_table_id": target_table_id,
        "target_table_name": target_table_id,
        "target_column_id": f"{target_table_id}-column",
        "target_column_name": "id",
        "use_in_join_paths": True,
    }


def test_connected_components_keep_shared_parent_edges_together() -> None:
    edges = [
        _edge("cards", "sets"),
        _edge("set-translations", "sets"),
        _edge("foreign-data", "cards"),
        _edge("orders", "customers"),
    ]

    components = semantic_fk_roles._connected_components(edges)

    assert [len(component) for component in components] == [3, 1]


def test_agent_uses_bounded_tool_then_persists_known_edge(monkeypatch) -> None:
    edge = _edge("cards", "sets")
    actions = iter(
        [
            SemanticFkRoleAction(
                action=SemanticFkRoleActionType.TOOL,
                tool=SemanticFkRoleToolName.GET_TABLE_CONTEXT,
                arguments=SemanticFkRoleToolArguments(table_id="cards"),
            ),
            SemanticFkRoleAction(
                action=SemanticFkRoleActionType.COMPLETE,
                proposals=[
                    SemanticFkRoleProposal(
                        column_id="cards-column",
                        attribute_id="sets-attribute",
                        role="card belongs to card set",
                        how_to_use="Traverse from a card row to its owning set.",
                        when_to_use=["a card question needs set properties"],
                        avoid_when=["the question asks only for the card name"],
                        evidence_ids=["cards", "sets"],
                    )
                ],
            ),
        ]
    )
    tool = MagicMock(return_value={"ok": True, "result": {"name": "cards"}})
    update = MagicMock(return_value=True)
    monkeypatch.setattr(
        semantic_fk_roles,
        "invoke_with_structured_output",
        lambda *_args, **_kwargs: next(actions),
    )
    monkeypatch.setattr(semantic_fk_roles, "run_semantic_fk_role_tool", tool)
    monkeypatch.setattr(
        semantic_fk_roles,
        "_update_compiled_semantic_fk_guidance",
        update,
    )

    result = semantic_fk_roles._analyze_work_item(
        "card_games",
        [edge],
        [edge],
        object(),
    )

    assert result == 1
    tool.assert_called_once()
    update.assert_called_once_with(
        "cards-column",
        "sets-attribute",
        role="card belongs to card set",
        how_to_use="Traverse from a card row to its owning set.",
        when_to_use=["a card question needs set properties"],
        avoid_when=["the question asks only for the card name"],
    )


def test_agent_discards_proposal_for_unknown_edge(monkeypatch) -> None:
    edge = _edge("cards", "sets")
    monkeypatch.setattr(
        semantic_fk_roles,
        "_update_compiled_semantic_fk_guidance",
        MagicMock(return_value=True),
    )

    updated = semantic_fk_roles._persist_proposals(
        [edge],
        [
            SemanticFkRoleProposal(
                column_id="invented-column",
                attribute_id="invented-attribute",
                role="invented",
                how_to_use="invented",
            )
        ],
    )

    assert updated == 0
    semantic_fk_roles._update_compiled_semantic_fk_guidance.assert_not_called()


def test_relationship_neighborhood_tool_is_read_only_and_bounded() -> None:
    result = run_semantic_fk_role_tool(
        SemanticFkRoleToolName.GET_RELATIONSHIP_NEIGHBORHOOD,
        {"table_id": "sets"},
        all_edges=[
            _edge("cards", "sets"),
            _edge("set-translations", "sets"),
        ],
    )

    assert result["ok"] is True
    assert len(result["result"]["edges"]) == 2
