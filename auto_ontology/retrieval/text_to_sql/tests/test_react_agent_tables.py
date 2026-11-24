# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from auto_ontology.retrieval.text_to_sql.react.agent import tables_in_database


def test_the_submitted_database_keeps_its_tables_whatever_their_case() -> None:
    tables = [
        {"name": "orders", "database_name": "adventureworks"},
        {"name": "state_2014", "database_name": "census"},
        {"name": "legacy", "database_name": None},
        {"name": "blank", "database_name": ""},
    ]

    kept = tables_in_database(tables, "AdventureWorks")

    assert [t["name"] for t in kept] == ["orders", "legacy", "blank"]
