# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from auto_ontology.retrieval.text_to_sql.agents import role_selection
from auto_ontology.retrieval.text_to_sql.formatters_util import format_role_links
from auto_ontology.semantic.embed import _build_rows


def _row(column: str, description: str, target_table: str = "u") -> dict:
    return {
        "source_table_id": "p",
        "source_table_name": "posts",
        "source_column_id": f"p.{column}",
        "source_column_name": column,
        "source_column_description": description,
        "target_table_id": target_table,
        "target_table_name": "users",
        "target_column_id": "u.Id",
        "target_column_name": "Id",
    }


def test_lists_every_role_towards_one_target():
    rows = [_row("OwnerUserId", "the owner"), _row("LastEditorUserId", "the editor")]

    text = format_role_links(rows, {"p", "u"})

    assert "posts -> users.Id:" in text
    assert "  - posts.OwnerUserId: the owner" in text
    assert "  - posts.LastEditorUserId: the editor" in text


def test_a_single_link_is_not_a_role_choice():
    assert format_role_links([_row("OwnerUserId", "the owner")], {"p", "u"}) == ""


def test_a_target_outside_the_prompt_is_left_out():
    rows = [_row("OwnerUserId", "a"), _row("LastEditorUserId", "b")]
    assert format_role_links(rows, {"p"}) == ""


ROLES = [
    _row("OwnerUserId", "the id of the owner user"),
    _row("LastEditorUserId", "the id of the last editor"),
]


def test_choose_role_follows_the_question_words():
    assert role_selection.choose_role("posts owned by Jay", ROLES) == "OwnerUserId"
    assert (
        role_selection.choose_role("posts edited by Jay", ROLES) == "LastEditorUserId"
    )


def test_choose_role_abstains_without_a_role_word():
    assert (
        role_selection.choose_role("how many posts did user Jay write", ROLES) is None
    )


def test_apply_question_roles_rewrites_the_hop(monkeypatch):
    monkeypatch.setattr(
        role_selection, "fetch_semantic_fk_join_groups", lambda _db: ROLES
    )
    paths = [
        {
            "path": [
                {
                    "source_database": "db",
                    "source_table": "posts",
                    "source_column": "LastEditorUserId",
                    "target_table": "users",
                    "target_column": "Id",
                }
            ]
        }
    ]

    assert role_selection.apply_question_roles("Which user owns post 5?", paths) == 1
    assert paths[0]["path"][0]["source_column"] == "OwnerUserId"


def test_profile_summary_is_embedded():
    rows = _build_rows(
        "db",
        {},
        [
            {
                "name": "Code",
                "term_name": "School",
                "id": "a1",
                "profile": '{"summary": "Zero-padded county code.", "row_count": 3}',
            }
        ],
    )

    assert "Zero-padded county code." in rows[0]["text"]
    assert "row_count" not in rows[0]["text"]


def test_many_roles_are_listed_by_name():
    rows = [_row(f"player_{i}", f"slot {i}") for i in range(8)]

    text = format_role_links(rows, {"p", "u"})

    assert "  - posts: player_0, player_1" in text
    assert "slot" not in text
