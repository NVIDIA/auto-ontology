# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Alembic environment for the GSF catalog/semantic schema.

**Scoped to one schema, deliberately.** Three systems share this database:
Prisma owns ``frontend`` (and ``prisma db push`` reconciles drift there on every
deploy), ``langchain_postgres`` owns ``vdb``, and this owns ``public``. Without
``include_object`` filtering, autogenerate would see the other two as "not in my
metadata" and cheerfully emit ``DROP TABLE`` for the user table and the vector
store. That is the single most destructive mistake available here, so the filter
is not an optimisation.

The connection URL comes from ``POSTGRES_*`` via the same helper the rest of the
codebase uses, rather than ``alembic.ini`` — one source of truth, and no
credentials in a committed file.
"""

from __future__ import annotations

import logging
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from gsf.dal.schema import METADATA
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

logger = logging.getLogger(__name__)

target_metadata = METADATA

#: Sentinel for "this object did not carry a schema at all", distinct from the
#: schema *being* None (which means the default schema, i.e. ours).
_MISSING = object()


def include_object(obj, name, type_, reflected, compare_to) -> bool:
    """Consider only objects in the connection's default schema.

    This is what keeps autogenerate away from the two schemas GSF does not own:
    Prisma's ``frontend`` and langchain_postgres' ``vdb``. Both are reflected and
    reach this function; without the filter they are tables Alembic can see but
    cannot find in ``METADATA``, and it would faithfully emit ``drop_table`` for
    every one of them.

    The corollary now that GSF holds ``public``: anything created there by
    something other than a migration *will* be proposed for deletion, because
    from here it is indistinguishable from a table someone removed from the
    model. That is the intended reading — ``public`` is GSF's — but it means a
    stray table is a loaded footgun rather than a curiosity.

    Alembic passes a wide range of objects through here — tables, columns,
    indexes, constraints — and they do not agree on how to reach their schema.
    Some carry ``.schema`` directly; some carry a ``.table`` that is itself a
    ``Table``; and some carry a ``.table`` that is only its *name*, as a string.
    Assuming the second shape raises ``'str' object has no attribute 'schema'``
    on the third.

    The default is to *exclude*: an object whose schema cannot be established is
    not ours, and letting it through risks autogenerate proposing a drop against
    somebody else's tables — the one outcome this filter exists to prevent.
    """
    # `_MISSING`, not None: None is a *meaningful* schema here -- it is how both
    # SQLAlchemy and Alembic spell "the default schema", which is ours. Using
    # None as the not-found marker too would make an object with no schema at all
    # indistinguishable from one explicitly in `public`.
    schema = getattr(obj, "schema", _MISSING)
    if schema is _MISSING:
        parent = getattr(obj, "table", None)
        schema = getattr(parent, "schema", _MISSING)
    if schema is _MISSING:
        # Constraints and indexes reached before their parent is resolved fall
        # here. Keep them only if the migration context is already scoped to us.
        return type_ not in {"table", "column"}
    # Both sides spell the default schema as None: the MetaData is unqualified,
    # and Alembic normalises a reflected default-schema table to None too. The
    # schemas GSF does not own -- `frontend` and `vdb` -- come back named, so
    # this single test is the whole filter.
    return schema is None



def _configure(**kwargs) -> None:
    context.configure(
        target_metadata=target_metadata,
        include_object=include_object,
        include_schemas=True,
        version_table="alembic_version",
        # Deliberately no `version_table_schema`. Naming the default schema while the
        # MetaData is unqualified makes Alembic fail to recognise its own version
        # table during autogenerate -- reflection reports the default schema as
        # None, the configured value says "public", they do not match, and the
        # table is treated as drift. The generated migration then contained
        # `op.drop_table('alembic_version')` in *upgrade*, i.e. the migration
        # deletes the record of which migrations have run. Left unset it resolves
        # through the connection's search_path, to the same place the tables go.
        #
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
        # Pin the role's search_path to `public`.
        #
        # This is the one place a schema name is still written down, and it is
        # deliberate: it is not a parameter the model reads, it is an assertion
        # about the *role*, making the default schema deterministic instead of
        # inherited. Everything else -- the MetaData, the migration DDL, every
        # raw query and the view -- is unqualified and simply resolves here.
        #
        # The hazard this defends against: the default search_path is
        # `"$user", public`, and `"$user"` is normally inert because no schema is
        # named after the role. Create one -- a schema named `gsf` while the role
        # is also called `gsf` -- and every unqualified `CREATE TABLE` in the
        # database silently retargets to it. That is not hypothetical; it is what
        # happened when GSF's tables lived in a `gsf` schema. Prisma's tables were
        # created there instead of its own schema, so the next `db push` found its
        # schema empty, tried to create them again, and failed with
        # `relation "conversations" already exists`, leaving the frontend blocked
        # forever on its wait-for-schema probe.
        #
        # Both of today's other owners are already immune by being explicit:
        # Prisma names `?schema=frontend` in its connection URL and
        # langchain_postgres is handed `vdb`. The pin is for whoever comes next
        # and will not have thought about any of this.
        #
        # Unconditional, and safe: `public` is the default anyway. Guarding on a
        # detected condition proved unreliable -- the check evaluated inside the
        # same transaction as the CREATE SCHEMA and did not fire -- and always
        # pinning costs nothing.
        connection.exec_driver_sql(
            "DO $$ BEGIN "
            # `%%I`, not `%I`: psycopg treats a bare % as a parameter placeholder
            # and rejects the statement before Postgres ever sees it.
            "EXECUTE format('ALTER ROLE %%I SET search_path TO public', current_user); "
            "EXCEPTION WHEN insufficient_privilege THEN NULL; "
            "END $$;"
        )

        connection.commit()

        _configure(connection=connection)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
