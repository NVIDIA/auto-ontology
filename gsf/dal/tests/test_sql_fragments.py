# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The description fallback.

Small enough to look obvious and consequential enough to be worth pinning: it
decides what the UI and the SQL generator see as a column's description, and
several of the semantic reads depend on it.

Three things it has to get right, each of which fails silently if wrong:

* **precedence** — own description, then ``HAS_ATTRIBUTE``, then
  ``SEMANTIC_FK``, in that order and not alphabetically;
* **blank counts as missing** — an empty string saved and cleared on a column
  must not mask a real description on the attribute behind it;
* **no fallback available** yields ``NULL``, not an empty string.
"""

from __future__ import annotations

import os
import uuid

import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import select  # noqa: E402

from gsf.dal import schema as s  # noqa: E402
from gsf.dal.session import store  # noqa: E402
from gsf.dal.sql_fragments import (  # noqa: E402
    column_description_expr,
    table_description_expr,
)


@pytest.fixture(scope="module", autouse=True)
def require_schema():
    if not os.environ.get("POSTGRES_USER"):
        pytest.skip("POSTGRES_* not set")
    try:
        store().query_read(f"SELECT 1 FROM {s.SCHEMA}.column_attribute LIMIT 1")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"gsf schema unavailable (alembic upgrade head): {exc}")


def _insert(table, **values):
    return store().query_write(table.insert().values(**values).returning(table.c.id))[
        0
    ]["id"]


@pytest.fixture
def fixture():
    """One column and one table, each with nothing described yet."""
    prefix = f"d-{uuid.uuid4().hex[:8]}"
    db = _insert(s.catalog_database, name=prefix)
    schema = _insert(s.catalog_schema, database_id=db, name="shop")
    table = _insert(s.catalog_table, schema_id=schema, name="customer")
    column = _insert(s.catalog_column, table_id=table, name="email")

    yield {"prefix": prefix, "table": table, "column": column}

    store().query_write(
        s.catalog_database.delete().where(s.catalog_database.c.name == prefix)
    )
    store().query_write(s.term.delete().where(s.term.c.name.like(f"{prefix}%")))
    store().query_write(
        s.column_attribute.delete().where(s.column_attribute.c.name.like(f"{prefix}%"))
    )


def _attribute(prefix: str, description: str | None) -> str:
    return _insert(
        s.column_attribute,
        name=f"{prefix}-attr-{uuid.uuid4().hex[:6]}",
        description=description,
        source_column="email",
        term_name="Customer",
        table_id="",
    )


def _describe_column(column_id: str) -> str | None:
    rows = store().query_read(
        select(column_description_expr().label("d")).where(
            s.catalog_column.c.id == column_id
        )
    )
    return rows[0]["d"]


def _describe_table(table_id: str) -> str | None:
    rows = store().query_read(
        select(table_description_expr().label("d")).where(
            s.catalog_table.c.id == table_id
        )
    )
    return rows[0]["d"]


# --------------------------------------------------------------------------
# Column
# --------------------------------------------------------------------------


def test_no_description_anywhere_is_null(fixture) -> None:
    assert _describe_column(fixture["column"]) is None


def test_the_columns_own_description_wins(fixture) -> None:
    store().query_write(
        s.catalog_column.update()
        .where(s.catalog_column.c.id == fixture["column"])
        .values(description="from the column")
    )
    attr = _attribute(fixture["prefix"], "from the attribute")
    store().query_write(
        s.column_has_attribute.insert().values(
            column_id=fixture["column"], attribute_id=attr
        )
    )
    assert _describe_column(fixture["column"]) == "from the column"


def test_falls_back_to_has_attribute(fixture) -> None:
    attr = _attribute(fixture["prefix"], "from the attribute")
    store().query_write(
        s.column_has_attribute.insert().values(
            column_id=fixture["column"], attribute_id=attr
        )
    )
    assert _describe_column(fixture["column"]) == "from the attribute"


def test_has_attribute_beats_semantic_fk(fixture) -> None:
    """Precedence, and not alphabetical.

    An attribute the column *is* an instance of describes it better than one it
    merely *references*.
    """
    owned = _attribute(fixture["prefix"], "owned")
    referenced = _attribute(fixture["prefix"], "referenced")
    store().query_write(
        s.column_has_attribute.insert().values(
            column_id=fixture["column"], attribute_id=owned
        )
    )
    store().query_write(
        s.column_semantic_fk.insert().values(
            column_id=fixture["column"], attribute_id=referenced
        )
    )
    assert _describe_column(fixture["column"]) == "owned"


def test_falls_back_to_semantic_fk_when_that_is_all_there_is(fixture) -> None:
    referenced = _attribute(fixture["prefix"], "referenced")
    store().query_write(
        s.column_semantic_fk.insert().values(
            column_id=fixture["column"], attribute_id=referenced
        )
    )
    assert _describe_column(fixture["column"]) == "referenced"


def test_a_blank_description_does_not_mask_the_fallback(fixture) -> None:
    """Saved and cleared is the case that matters.

    An empty string is what a UI leaves behind when someone deletes the text.
    Treating it as present would make the fallback useless for exactly the rows
    a user has touched.
    """
    store().query_write(
        s.catalog_column.update()
        .where(s.catalog_column.c.id == fixture["column"])
        .values(description="   ")
    )
    attr = _attribute(fixture["prefix"], "from the attribute")
    store().query_write(
        s.column_has_attribute.insert().values(
            column_id=fixture["column"], attribute_id=attr
        )
    )
    assert _describe_column(fixture["column"]) == "from the attribute"


def test_a_blank_attribute_description_is_skipped(fixture) -> None:
    blank = _attribute(fixture["prefix"], "")
    real = _attribute(fixture["prefix"], "the real one")
    for attr in (blank, real):
        store().query_write(
            s.column_has_attribute.insert().values(
                column_id=fixture["column"], attribute_id=attr
            )
        )
    assert _describe_column(fixture["column"]) == "the real one"


def test_another_columns_attribute_is_not_borrowed(fixture) -> None:
    """The subquery correlates on column id; without that it would take any row."""
    other = _insert(s.catalog_column, table_id=fixture["table"], name="phone")
    attr = _attribute(fixture["prefix"], "belongs to phone")
    store().query_write(
        s.column_has_attribute.insert().values(column_id=other, attribute_id=attr)
    )
    assert _describe_column(fixture["column"]) is None


# --------------------------------------------------------------------------
# Table
# --------------------------------------------------------------------------


def test_table_falls_back_to_its_term(fixture) -> None:
    term = _insert(s.term, name=f"{fixture['prefix']}-Customer", description="a buyer")
    store().query_write(
        s.table_term.insert().values(table_id=fixture["table"], term_id=term)
    )
    assert _describe_table(fixture["table"]) == "a buyer"


def test_table_own_description_wins(fixture) -> None:
    store().query_write(
        s.catalog_table.update()
        .where(s.catalog_table.c.id == fixture["table"])
        .values(description="from the table")
    )
    term = _insert(s.term, name=f"{fixture['prefix']}-Customer", description="a buyer")
    store().query_write(
        s.table_term.insert().values(table_id=fixture["table"], term_id=term)
    )
    assert _describe_table(fixture["table"]) == "from the table"


def test_blank_table_description_does_not_mask_the_term(fixture) -> None:
    store().query_write(
        s.catalog_table.update()
        .where(s.catalog_table.c.id == fixture["table"])
        .values(description="")
    )
    term = _insert(s.term, name=f"{fixture['prefix']}-Customer", description="a buyer")
    store().query_write(
        s.table_term.insert().values(table_id=fixture["table"], term_id=term)
    )
    assert _describe_table(fixture["table"]) == "a buyer"


def test_table_with_no_term_is_null(fixture) -> None:
    assert _describe_table(fixture["table"]) is None
