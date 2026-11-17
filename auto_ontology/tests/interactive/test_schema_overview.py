# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from auto_ontology.retrieval.interactive.clarify import _schema_overview
from auto_ontology.retrieval.interactive.coordinator import create_session
from auto_ontology.retrieval.interactive.kb_coverage import (
    _compact_column_meanings,
    _compact_schema,
)

_PLAIN_DDL = """\
CREATE TABLE "orders" (
id text NOT NULL,
placed_at timestamp NULL,
    PRIMARY KEY (id)
);

First 3 rows:
id    placed_at
"""

_QUOTED_DDL = """\
"CREATE" TABLE "DataFlow" (
"RecordRegistry" text NOT NULL,
flow_overview jsonb NULL,
    "PRIMARY" KEY (RecordRegistry)
);

"First" 3 rows:
RecordRegistry    flow_overview

"CREATE" TABLE "RiskManagement" (
"RISKTRACE" bigint NOT NULL,
flow_link text NOT NULL,
    "PRIMARY" KEY (RISKTRACE),
    "FOREIGN" KEY (flow_link) REFERENCES DataFlow(RecordRegistry)
);
"""


def test_compact_schema_lists_columns_of_plain_ddl():
    assert _compact_schema(_PLAIN_DDL) == "orders: id, placed_at"


def test_compact_schema_reads_ddl_with_quoted_keywords():
    assert _compact_schema(_QUOTED_DDL) == (
        "DataFlow: RecordRegistry, flow_overview\nRiskManagement: RISKTRACE, flow_link"
    )


def test_compact_column_meanings_groups_columns_by_table():
    meanings = {
        "cross_border|DataFlow|RecordRegistry": "TEXT. Primary key.",
        "cross_border|DataFlow|flow_overview": {"column_meaning": "JSONB column."},
        "cross_border|RiskManagement|RISKTRACE": "BIGINT.",
    }
    assert _compact_column_meanings(json.dumps(meanings)) == (
        "DataFlow: RecordRegistry, flow_overview\nRiskManagement: RISKTRACE"
    )


def test_compact_column_meanings_ignores_unreadable_input():
    assert _compact_column_meanings("not json") == ""
    assert _compact_column_meanings("[]") == ""
    assert _compact_column_meanings(json.dumps({"no_pipes": "x"})) == ""


def _session(**kwargs):
    defaults = dict(
        session_id="s1",
        task_id="t1",
        db_name="cross_border",
        db_schema="",
        external_kb="[]",
        question="q",
        data_retriever=None,
        semantic_retriever=None,
        connectors=[],
    )
    defaults.update(kwargs)
    return create_session(**defaults)


def test_clarify_lists_columns_from_column_meanings_when_given():
    meanings = json.dumps({"cross_border|DataFlow|RecordRegistry": "TEXT."})
    session = _session(db_schema=_PLAIN_DDL, column_meanings=meanings)
    assert _schema_overview(session) == "DataFlow: RecordRegistry"


def test_clarify_falls_back_to_the_schema_without_column_meanings():
    assert _schema_overview(_session(db_schema=_PLAIN_DDL)) == "orders: id, placed_at"


def test_create_session_defaults_to_no_column_meanings():
    assert _session().column_meanings == ""
