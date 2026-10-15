# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Groups, membership, and the accounts they point at.

Needs a migrated database. The account table is Prisma's, so CI's Alembic
database does not have it; the fixture creates the columns this DAL reads
when that table is absent and leaves a real one untouched.
"""

from __future__ import annotations

from collections.abc import Iterator
import os
import uuid

import pytest
from sqlalchemy import text

pytest.importorskip("sqlalchemy")

from auto_ontology.dal import groups as dal  # noqa: E402
from auto_ontology.dal import schema as s  # noqa: E402
from auto_ontology.dal.session import store  # noqa: E402
from auto_ontology.infra.postgres import FRONTEND_SCHEMA  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def require_schema() -> Iterator[None]:
    if not os.environ.get("POSTGRES_USER"):
        pytest.skip("POSTGRES_* not set")
    store().query_read(text("SELECT 1 FROM user_group LIMIT 1"))

    schema_existed = bool(
        store().query_read(
            text("SELECT 1 FROM pg_namespace WHERE nspname = :schema"),
            {"schema": FRONTEND_SCHEMA},
        )
    )
    table_existed = bool(
        store().query_read(
            text(
                """
                SELECT 1
                  FROM pg_class AS class
                  JOIN pg_namespace AS namespace
                    ON namespace.oid = class.relnamespace
                 WHERE namespace.nspname = :schema
                   AND class.relname = 'user'
                """
            ),
            {"schema": FRONTEND_SCHEMA},
        )
    )

    store().query_write(text(f"CREATE SCHEMA IF NOT EXISTS {FRONTEND_SCHEMA}"))
    store().query_write(
        text(
            f"""
            CREATE TABLE IF NOT EXISTS {FRONTEND_SCHEMA}."user" (
                id text PRIMARY KEY,
                name text NOT NULL,
                email text NOT NULL,
                "emailVerified" boolean NOT NULL DEFAULT false,
                "createdAt" timestamptz NOT NULL DEFAULT now(),
                "updatedAt" timestamptz NOT NULL DEFAULT now(),
                role text,
                banned boolean NOT NULL DEFAULT false
            )
            """
        )
    )
    try:
        yield
    finally:
        if not table_existed:
            store().query_write(text(f'DROP TABLE IF EXISTS {FRONTEND_SCHEMA}."user"'))
        if not schema_existed:
            schema_objects = store().query_read(
                text(
                    """
                    SELECT 1
                      FROM pg_class AS class
                      JOIN pg_namespace AS namespace
                        ON namespace.oid = class.relnamespace
                     WHERE namespace.nspname = :schema
                     LIMIT 1
                    """
                ),
                {"schema": FRONTEND_SCHEMA},
            )
            if not schema_objects:
                store().query_write(text(f"DROP SCHEMA IF EXISTS {FRONTEND_SCHEMA}"))


@pytest.fixture
def accounts():
    """Two Better Auth accounts, removed with the groups this test creates."""
    prefix = f"grp-{uuid.uuid4().hex[:8]}"
    created: list[str] = []

    def make(*, role: str = "viewer", name: str | None = None) -> str:
        user_id = f"{prefix}-{len(created)}"
        store().query_write(
            text(
                f"""
                INSERT INTO {FRONTEND_SCHEMA}."user"
                    (
                        id, name, email, "emailVerified", "createdAt", "updatedAt",
                        role, banned
                    )
                VALUES
                    (:id, :name, :email, true, now(), now(), :role, false)
                """
            ),
            {
                "id": user_id,
                "name": name or user_id,
                "email": f"{user_id}@example.com",
                "role": role,
            },
        )
        created.append(user_id)
        return user_id

    yield make

    store().query_write(
        s.user_group.delete().where(s.user_group.c.name.like(f"{prefix}%"))
    )
    if created:
        store().query_write(
            text(f'DELETE FROM {FRONTEND_SCHEMA}."user" WHERE id = ANY(:ids)'),
            {"ids": created},
        )


def test_a_group_keeps_its_members(accounts) -> None:
    first = accounts(name="Ada")
    second = accounts(name="Grace", role="admin")
    prefix = first.rsplit("-", 1)[0]

    created = dal.create_group(
        name=f"{prefix} Finance",
        description="books",
        color="#76b900",
        role="viewer",
        user_ids=[second, first],
    )

    assert created["description"] == "books"
    assert created["role"] == "viewer"
    assert [user["name"] for user in created["users"]] == ["Ada", "Grace"]
    assert created["users"][0]["role"] == "viewer"
    assert created["users"][1]["role"] == "admin"

    again = dal.get_group(created["id"])
    assert again is not None
    assert [user["id"] for user in again["users"]] == [first, second]


def test_names_collide_case_insensitively(accounts) -> None:
    user_id = accounts()
    prefix = user_id.rsplit("-", 1)[0]
    dal.create_group(
        name=f"{prefix} Finance",
        description=None,
        color="#000",
        role="admin",
        user_ids=[user_id],
    )
    with pytest.raises(dal.GroupNameTaken):
        dal.create_group(
            name=f"  {prefix} finance  ",
            description=None,
            color="#111",
            role="viewer",
            user_ids=[],
        )


def test_an_unknown_member_is_refused(accounts) -> None:
    user_id = accounts()
    prefix = user_id.rsplit("-", 1)[0]
    with pytest.raises(dal.UnknownGroupMembers) as refused:
        dal.create_group(
            name=f"{prefix} Finance",
            description=None,
            color="#000",
            role="viewer",
            user_ids=[user_id, "nobody"],
        )
    assert refused.value.user_ids == ["nobody"]


def test_replacing_members_and_deleting_the_group(accounts) -> None:
    first = accounts(name="Ada")
    second = accounts(name="Grace")
    prefix = first.rsplit("-", 1)[0]
    created = dal.create_group(
        name=f"{prefix} Finance",
        description=None,
        color="#000",
        role="viewer",
        user_ids=[first],
    )

    updated = dal.update_group(
        group_id=created["id"],
        updates={"role": "admin"},
        user_ids=[second],
    )
    assert updated is not None
    assert updated["role"] == "admin"
    assert [user["id"] for user in updated["users"]] == [second]
    assert [group["id"] for group in dal.groups_for_user(second)] == [created["id"]]
    assert dal.groups_for_user(first) == []

    assert dal.delete_group(created["id"]) is True
    assert dal.get_group(created["id"]) is None
    assert dal.delete_group(created["id"]) is False


def test_directory_lists_accounts_with_their_group_ids(accounts) -> None:
    first = accounts(name="Ada")
    second = accounts(name="Grace")
    prefix = first.rsplit("-", 1)[0]
    finance = dal.create_group(
        name=f"{prefix} Finance",
        description=None,
        color="#000",
        role="viewer",
        user_ids=[first],
    )
    legal = dal.create_group(
        name=f"{prefix} Legal",
        description=None,
        color="#111",
        role="admin",
        user_ids=[first, second],
    )

    directory = dal.list_users_and_groups()
    by_id = {user["id"]: user for user in directory["users"]}
    assert by_id[first]["groups"] == [finance["id"], legal["id"]]
    assert by_id[second]["groups"] == [legal["id"]]


def test_updating_a_missing_group_does_not_write_members(accounts) -> None:
    user_id = accounts()
    assert (
        dal.update_group(
            group_id="missing-group",
            updates={"color": "#fff"},
            user_ids=[user_id],
        )
        is None
    )
    assert dal.get_group("missing-group") is None


def test_a_deleted_account_stays_visible_on_the_group(accounts) -> None:
    user_id = accounts(name="Ada")
    prefix = user_id.rsplit("-", 1)[0]
    created = dal.create_group(
        name=f"{prefix} Finance",
        description=None,
        color="#000",
        role="viewer",
        user_ids=[user_id],
    )
    store().query_write(
        text(f'DELETE FROM {FRONTEND_SCHEMA}."user" WHERE id = :id'),
        {"id": user_id},
    )

    group = dal.get_group(created["id"])
    assert group is not None
    assert group["users"] == [{"id": user_id, "name": "", "email": "", "role": None}]
