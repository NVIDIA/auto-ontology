# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Alembic environment for the GSF catalog/semantic schema.

**Scoped to the ``gsf`` schema, deliberately.** Three systems share this
database: Prisma owns ``public`` (and ``prisma db push`` reconciles drift there
on every deploy), ``langchain_postgres`` owns ``vdb``, and this owns ``gsf``.
Without ``include_object`` filtering, autogenerate would see Prisma's tables as
"not in my metadata" and cheerfully emit ``DROP TABLE`` for the user table.
That is the single most destructive mistake available here, so the filter is
not an optimisation.

The connection URL comes from ``POSTGRES_*`` via the same helper the rest of the
codebase uses, rather than ``alembic.ini`` — one source of truth, and no
credentials in a committed file.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from gsf.dal.schema import METADATA, SCHEMA
from gsf.dal.session import sqlalchemy_url
from gsf.env import load_env

load_env()

config = context.config
# `%` doubled: set_main_option goes through ConfigParser, which applies
# pyformat interpolation, so a password containing a percent sign raises
# "invalid interpolation syntax" and alembic dies before running anything --
# on a password SQLAlchemy itself accepts.
config.set_main_option("sqlalchemy.url", sqlalchemy_url().replace("%", "%%"))

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = METADATA


def include_object(obj, name, type_, reflected, compare_to) -> bool:
    """Consider only objects in the ``gsf`` schema.

    Alembic passes a wide range of objects through here — tables, columns,
    indexes, constraints — and they do not agree on how to reach their schema.
    Some carry ``.schema`` directly; some carry a ``.table`` that is itself a
    ``Table``; and some carry a ``.table`` that is only its *name*, as a string.
    Assuming the second shape raises ``'str' object has no attribute 'schema'``
    on the third.

    The default is to *exclude*: an object whose schema cannot be established is
    not ours, and letting it through risks autogenerate proposing a drop against
    Prisma's tables — the one outcome this filter exists to prevent.
    """
    schema = getattr(obj, "schema", None)
    if schema is None:
        parent = getattr(obj, "table", None)
        schema = getattr(parent, "schema", None)
    if schema is None:
        # Constraints and indexes reached before their parent is resolved fall
        # here. Keep them only if the migration context is already scoped to us.
        return type_ not in {"table", "column"}
    return schema == SCHEMA


def _configure(**kwargs) -> None:
    context.configure(
        target_metadata=target_metadata,
        include_object=include_object,
        include_schemas=True,
        version_table="alembic_version",
        version_table_schema=SCHEMA,
        # Without this a changed column type is silently ignored by
        # autogenerate, which is worse than a false positive.
        compare_type=True,
        compare_server_default=True,
        **kwargs,
    )


def run_migrations_offline() -> None:
    _configure(
        url=config.get_main_option("sqlalchemy.url"),
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        # The version table lives in `gsf`, so the schema has to exist before
        # Alembic tries to stamp anything -- including on a completely blank
        # database, where migration 0001 has not run yet.
        connection.exec_driver_sql(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
        connection.commit()

        _configure(connection=connection)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
