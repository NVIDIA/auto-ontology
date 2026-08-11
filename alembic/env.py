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

from gsf.dal.pg.schema import METADATA, SCHEMA
from gsf.dal.pg.session import sqlalchemy_url
from gsf.env import load_env

load_env()

config = context.config
config.set_main_option("sqlalchemy.url", sqlalchemy_url())

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = METADATA


def include_object(obj, name, type_, reflected, compare_to) -> bool:
    """Consider only objects in the ``gsf`` schema.

    ``reflected`` objects carry the schema they were found in; objects from our
    own metadata carry ``SCHEMA`` because ``MetaData(schema=...)`` sets it. Both
    paths are checked, so neither a stray reflected table nor a mis-declared one
    slips through.
    """
    if type_ == "table":
        return obj.schema == SCHEMA
    if hasattr(obj, "table"):
        return obj.table.schema == SCHEMA
    return True


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
