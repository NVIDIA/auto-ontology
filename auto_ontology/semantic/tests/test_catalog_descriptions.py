# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from auto_ontology.semantic import catalog_descriptions as cd
from auto_ontology.server.datasources import service as datasources_service


def _col(cid: str, name: str, description: str = "", **extra: Any) -> dict:
    return {
        "id": cid,
        "name": name,
        "data_type": "varchar",
        "sample_values": [],
        "description": description,
        "own_description": description,
        "references": None,
        **extra,
    }


_DESCRIBED = {
    "id": "t-dept",
    "name": "departments",
    "schema_name": "hr",
    "description": "A department.",
    "own_description": "A department.",
    "bridge_description": None,
    "columns": [_col("c-dept-key", "dept_key", "Department key.")],
}
_ROLES = {
    "id": "t-roles",
    "name": "roles",
    "schema_name": "hr",
    "description": None,
    "own_description": None,
    "bridge_description": None,
    "columns": [
        _col(
            "c-roles-dept",
            "dept_key",
            "Department key.",
            own_description=None,
            references="hr.departments.dept_key",
        ),
        _col("c-roles-user", "username", sample_values=["jdoe", "asmith"]),
    ],
}


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    seen: dict[str, Any] = {"prompts": [], "filled": None, "refreshed": None}

    def _invoke(llm: Any, messages: list, schema: type) -> Any:
        seen["prompts"].append(messages[-1].content)
        return schema(
            table_description="Which user holds a role in which department.",
            columns=[
                cd.ColumnDescription(column_name="username", description="User login."),
                cd.ColumnDescription(column_name="dept_key", description="Ignored."),
            ],
        )

    def _fill(tables: dict, columns: dict) -> list[str]:
        seen["filled"] = (tables, columns)
        return [*tables, *columns]

    monkeypatch.setattr(
        cd, "fetch_description_context", lambda db: [_DESCRIBED, _ROLES]
    )
    monkeypatch.setattr(cd, "get_llm_client", lambda **kw: object())
    monkeypatch.setattr(cd, "invoke_with_structured_output", _invoke)
    monkeypatch.setattr(cd, "fill_empty_descriptions", _fill)
    monkeypatch.setattr(cd, "fetch_catalog_node_ids", lambda db: ["t-dept", "t-roles"])
    monkeypatch.setattr(
        datasources_service,
        "refresh_catalog_embeddings",
        lambda ids: seen.__setitem__("refreshed", ids),
    )
    return seen


def test_only_what_has_no_description_of_its_own_is_written(
    world: dict[str, Any],
) -> None:
    assert cd.describe_catalog("hr") == 3

    tables, columns = world["filled"]
    assert tables == {"t-roles": "Which user holds a role in which department."}
    assert columns == {"c-roles-dept": "Department key.", "c-roles-user": "User login."}


def test_a_borrowed_description_is_copied_not_rewritten(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cd, "invoke_with_structured_output", lambda *a: None)

    assert cd.describe_catalog("hr") == 1
    assert world["filled"] == ({}, {"c-roles-dept": "Department key."})


def test_one_call_per_table_that_needs_one(world: dict[str, Any]) -> None:
    cd.describe_catalog("hr")

    assert len(world["prompts"]) == 1
    prompt = world["prompts"][0]
    assert "Table: hr.roles" in prompt
    assert (
        "dept_key (varchar), foreign key to hr.departments.dept_key: Department key."
        in prompt
    )
    assert (
        "username (varchar), sample values: jdoe, asmith: (needs a description)"
        in prompt
    )


def test_the_whole_catalog_is_re_embedded(world: dict[str, Any]) -> None:
    cd.describe_catalog("hr")

    assert world["refreshed"] == ["t-dept", "t-roles"]


def test_a_failed_call_leaves_the_table_undescribed(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cd, "invoke_with_structured_output", lambda *a: None)

    cd.describe_catalog("hr")
    tables, columns = world["filled"]
    assert "t-roles" not in tables and "c-roles-user" not in columns
    assert world["refreshed"] == ["t-dept", "t-roles"]
