# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User groups and the accounts that belong to them.

Accounts themselves stay in Prisma's ``frontend."user"`` table. This module
reads that table and writes only ``user_group`` / ``user_group__member``,
which Alembic owns. A membership stores the Better Auth id with no foreign
key — see the comment on those tables in ``auto_ontology/dal/schema.py``.
"""

from __future__ import annotations

from typing import Any, NoReturn

from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError

from auto_ontology.dal import schema as s
from auto_ontology.dal.session import store, write_transaction
from auto_ontology.infra.postgres import FRONTEND_SCHEMA

_GROUP_COLUMNS = (
    s.user_group.c.id,
    s.user_group.c.name,
    s.user_group.c.description,
    s.user_group.c.color,
    s.user_group.c.role,
    s.user_group.c.created,
)

_ACCOUNT_SQL = f"""
SELECT id, name, email, role
  FROM {FRONTEND_SCHEMA}."user"
"""


class GroupNameTaken(ValueError):
    """Another group already holds this name, case-insensitively."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"Group with name {name!r} already exists")


class UnknownGroupMembers(ValueError):
    """One or more member ids are not accounts."""

    def __init__(self, user_ids: list[str]) -> None:
        self.user_ids = user_ids
        listed = ", ".join(user_ids)
        super().__init__(f"Unknown user id(s): {listed}")


def _require_role(role: str) -> None:
    if role not in (s.GROUP_ROLE_ADMIN, s.GROUP_ROLE_VIEWER):
        raise ValueError(f"Role must be admin or viewer, got {role!r}")


def _name_taken(name: str, *, exclude_id: str | None = None) -> bool:
    statement = select(s.user_group.c.id).where(
        func.lower(func.trim(s.user_group.c.name)) == name.strip().lower()
    )
    if exclude_id is not None:
        statement = statement.where(s.user_group.c.id != exclude_id)
    return bool(store().query_read(statement))


def _accounts_by_id(user_ids: list[str]) -> dict[str, dict[str, Any]]:
    if not user_ids:
        return {}
    rows = store().query_read(
        text(_ACCOUNT_SQL + " WHERE id = ANY(:user_ids)"),
        {"user_ids": list(user_ids)},
    )
    return {row["id"]: dict(row) for row in rows}


def _missing_user_ids(user_ids: list[str]) -> list[str]:
    found = _accounts_by_id(user_ids)
    return [user_id for user_id in user_ids if user_id not in found]


def _user_view(account: dict[str, Any] | None, user_id: str) -> dict[str, Any]:
    """An account as a group member, or the bare id when the account is gone."""
    if account is None:
        return {"id": user_id, "name": "", "email": "", "role": None}
    return {
        "id": account["id"],
        "name": account["name"] or "",
        "email": account["email"] or "",
        "role": account["role"],
    }


def _attach_users(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not groups:
        return []
    group_ids = [group["id"] for group in groups]
    member_rows = store().query_read(
        select(
            s.user_group__member.c.group_id,
            s.user_group__member.c.user_id,
        ).where(s.user_group__member.c.group_id.in_(group_ids))
    )
    accounts = _accounts_by_id(list({row["user_id"] for row in member_rows}))
    members: dict[str, list[str]] = {}
    for row in member_rows:
        members.setdefault(row["group_id"], []).append(row["user_id"])

    attached: list[dict[str, Any]] = []
    for group in groups:
        users = [
            _user_view(accounts.get(user_id), user_id)
            for user_id in members.get(group["id"], [])
        ]
        users.sort(key=lambda user: ((user["name"] or "").lower(), user["id"]))
        attached.append(
            {
                **group,
                "description": group["description"] or "",
                "users": users,
            }
        )
    return attached


def _group_rows(
    *, group_id: str | None = None, user_id: str | None = None
) -> list[dict[str, Any]]:
    statement = select(*_GROUP_COLUMNS).order_by(
        func.lower(s.user_group.c.name), s.user_group.c.id
    )
    if user_id is not None:
        statement = statement.join(
            s.user_group__member,
            s.user_group__member.c.group_id == s.user_group.c.id,
        ).where(s.user_group__member.c.user_id == user_id)
    if group_id is not None:
        statement = statement.where(s.user_group.c.id == group_id)
    return _attach_users(store().query_read(statement))


def _insert_members(group_id: str, user_ids: list[str]) -> None:
    if not user_ids:
        return
    store().query_write(
        s.user_group__member.insert().values(
            [{"group_id": group_id, "user_id": user_id} for user_id in user_ids]
        )
    )


def _replace_members(group_id: str, user_ids: list[str]) -> None:
    store().query_write(
        s.user_group__member.delete().where(s.user_group__member.c.group_id == group_id)
    )
    _insert_members(group_id, user_ids)


def _raise_name_taken(exc: IntegrityError, name: str) -> NoReturn:
    if "uq_user_group_name_lower" not in str(exc.orig):
        raise exc
    raise GroupNameTaken(name) from exc


def list_groups() -> list[dict[str, Any]]:
    """Every group, ordered by name, each with its members."""
    return _group_rows()


def get_group(group_id: str) -> dict[str, Any] | None:
    """One group with its members, or ``None`` when the id is unknown."""
    rows = _group_rows(group_id=group_id)
    return rows[0] if rows else None


def groups_for_user(user_id: str) -> list[dict[str, Any]]:
    """The groups *user_id* belongs to, ordered by name.

    Does not check that the account exists. A deleted account can still have
    memberships, and those are part of the answer.
    """
    return _group_rows(user_id=user_id)


def get_account(user_id: str) -> dict[str, Any] | None:
    """One Better Auth account, or ``None`` when the id is unknown."""
    rows = store().query_read(
        text(_ACCOUNT_SQL + " WHERE id = :user_id"),
        {"user_id": user_id},
    )
    if not rows:
        return None
    account = rows[0]
    return _user_view(dict(account), account["id"])


def list_users_and_groups() -> dict[str, Any]:
    """Every account, and every group.

    Each account carries the ids of the groups it belongs to, in group-name
    order. A membership whose account has been deleted stays on the group and
    does not appear in ``users``.
    """
    accounts = store().query_read(text(_ACCOUNT_SQL + " ORDER BY lower(name), id"))
    groups = list_groups()
    group_ids: dict[str, list[str]] = {}
    for group in groups:
        for user in group["users"]:
            group_ids.setdefault(user["id"], []).append(group["id"])
    users = [
        {
            **_user_view(dict(account), account["id"]),
            "groups": group_ids.get(account["id"], []),
        }
        for account in accounts
    ]
    return {"users": users, "groups": groups}


def create_group(
    *,
    name: str,
    description: str | None,
    color: str,
    role: str,
    user_ids: list[str],
) -> dict[str, Any]:
    """Create a group and its membership, and return the group.

    *name* is stored as given; the caller trims it. Raises
    :class:`GroupNameTaken` when the name is already held, and
    :class:`UnknownGroupMembers` when a member id is not an account.
    """
    _require_role(role)
    if _name_taken(name):
        raise GroupNameTaken(name)
    missing = _missing_user_ids(user_ids)
    if missing:
        raise UnknownGroupMembers(missing)

    try:
        with write_transaction():
            rows = store().query_write(
                s.user_group.insert()
                .values(
                    name=name,
                    description=description,
                    color=color,
                    role=role,
                )
                .returning(s.user_group.c.id)
            )
            _insert_members(rows[0]["id"], user_ids)
            group_id = rows[0]["id"]
    except IntegrityError as exc:
        _raise_name_taken(exc, name)

    group = get_group(group_id)
    if group is None:  # pragma: no cover - the insert just returned this id
        raise RuntimeError(f"Group {group_id!r} missing after insert")
    return group


def update_group(
    *,
    group_id: str,
    updates: dict[str, Any],
    user_ids: list[str] | None,
) -> dict[str, Any] | None:
    """Update a group's fields and, when given, replace its members.

    ``None`` when no group has that id. *user_ids* ``None`` leaves membership
    alone; a list replaces it, including the empty list.

    The group row is locked for the write. A delete that lands first comes
    back as ``None`` instead of a foreign-key error, and two membership
    replacements of the same group run one after the other.

    Raises :class:`GroupNameTaken` and :class:`UnknownGroupMembers` for the
    same reasons as :func:`create_group`. A group's own name is not a
    collision.
    """
    if "role" in updates:
        _require_role(updates["role"])
    if user_ids is not None:
        missing = _missing_user_ids(user_ids)
        if missing:
            raise UnknownGroupMembers(missing)

    try:
        with write_transaction():
            locked = store().query_read(
                select(s.user_group.c.id)
                .where(s.user_group.c.id == group_id)
                .with_for_update()
            )
            if not locked:
                return None
            if "name" in updates and _name_taken(updates["name"], exclude_id=group_id):
                raise GroupNameTaken(updates["name"])
            if updates:
                store().query_write(
                    s.user_group.update()
                    .where(s.user_group.c.id == group_id)
                    .values(**updates)
                )
            if user_ids is not None:
                _replace_members(group_id, user_ids)
    except IntegrityError as exc:
        if "name" in updates:
            _raise_name_taken(exc, updates["name"])
        raise

    return get_group(group_id)


def delete_group(group_id: str) -> bool:
    """Delete one group. Membership goes with it. ``False`` when it is absent."""
    rows = store().query_write(
        s.user_group.delete()
        .where(s.user_group.c.id == group_id)
        .returning(s.user_group.c.id)
    )
    return bool(rows)
