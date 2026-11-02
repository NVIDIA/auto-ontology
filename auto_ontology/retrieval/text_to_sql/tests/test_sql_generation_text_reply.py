# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reading an SQLGenerationModel out of a reply that skipped the tool call.

The three layouts are the ones Claude used when it answered in text, taken from
captured benchmark calls.
"""

import pytest

from auto_ontology.retrieval.text_to_sql.models import SQLGenerationModel

FENCED = """**thought:** The earlier query returned nothing, so I join on the term.

**sql_code:**
```sql
SELECT t.TERM_CODE
FROM ACADEMIC_TERMS t
```

**response:** This lists every academic term code."""

TAGGED = """<thought>
Kept the department filter.
</thought>
<sql_code>
WITH d AS (SELECT 1 AS x) SELECT x FROM d
</sql_code>
<response>
Shows one value per department.
</response>"""

LABELLED = """**thought**: Math is department code '18'.

**sql_code**: SELECT DEPARTMENT_NAME FROM SIS_DEPARTMENT WHERE DEPARTMENT_CODE = '18'

**response**: Shows the Mathematics department name."""


@pytest.mark.parametrize(
    "text, sql, thought, response",
    [
        (
            FENCED,
            "SELECT t.TERM_CODE\nFROM ACADEMIC_TERMS t",
            "The earlier query returned nothing, so I join on the term.",
            "This lists every academic term code.",
        ),
        (
            TAGGED,
            "WITH d AS (SELECT 1 AS x) SELECT x FROM d",
            "Kept the department filter.",
            "Shows one value per department.",
        ),
        (
            LABELLED,
            "SELECT DEPARTMENT_NAME FROM SIS_DEPARTMENT WHERE DEPARTMENT_CODE = '18'",
            "Math is department code '18'.",
            "Shows the Mathematics department name.",
        ),
    ],
    ids=["fenced", "tagged", "labelled"],
)
def test_each_text_layout_yields_the_three_fields(text, sql, thought, response):
    result = SQLGenerationModel.from_text_reply(text)

    assert result is not None
    assert result.sql_code == sql
    assert result.thought == thought
    assert result.response == response


def test_the_last_fenced_block_is_the_query():
    text = "Before:\n```sql\nSELECT 1\n```\nCorrected:\n```sql\nSELECT 2\n```\nDone."

    assert SQLGenerationModel.from_text_reply(text).sql_code == "SELECT 2"


@pytest.mark.parametrize(
    "text",
    [
        "The earlier query was wrong because the join dropped rows.",
        "```\nThis block is prose, not a query.\n```",
    ],
    ids=["no-sql", "fenced-prose"],
)
def test_text_without_a_query_is_not_an_answer(text):
    assert SQLGenerationModel.from_text_reply(text) is None


def test_a_bare_query_still_gets_a_thought_and_response():
    result = SQLGenerationModel.from_text_reply("```sql\nSELECT 1\n```")

    assert result.sql_code == "SELECT 1"
    assert result.thought and result.response
