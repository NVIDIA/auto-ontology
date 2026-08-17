# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backfill embeddings from the vdb schema, then drop it

The vectors in ``vdb.data_objects_layer`` / ``vdb.semantic_layer`` are already
one-to-one with the rows they describe: each carries its entity's own id in
``langchain_metadata->>'id'``. That was verified before this was written --
51/51 Tables and 589/589 Columns joined their catalog row exactly -- which is
what makes a copy possible instead of a re-embed.

Re-embedding would also have worked (nothing is released), but it costs a full
pass over the embedding model for every deployment, and a copy is exact.

**Idempotent and optional.** A database created after this change has no ``vdb``
schema at all, so every statement is guarded on its existence. Nothing here
fails on a fresh install; it simply does nothing.

**Not reversible.** ``downgrade`` recreates neither the schema nor its contents:
the embeddings still exist, on the entity rows, which is where the application
now reads them. Recreating the old collections would mean re-embedding, and a
downgrade that silently costs an inference run is worse than one that declines.

Revision ID: a98e6e7ee627
Revises: ff032ef08929
Create Date: 2026-08-17 15:20:00.000000

"""

from typing import Sequence, Union

from alembic import op

revision: str = "a98e6e7ee627"
down_revision: Union[str, Sequence[str], None] = "ff032ef08929"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


#: (source collection, label, destination table).
#:
#: The labels are the ones actually observed in a populated store, not every
#: label the code can emit -- a label with no rows copies nothing, so listing it
#: would only be noise.
_BACKFILL: tuple[tuple[str, str, str], ...] = (
    ("data_objects_layer", "Table", "catalog_table"),
    ("data_objects_layer", "Column", "catalog_column"),
    ("semantic_layer", "Term", "term"),
    ("semantic_layer", "ColumnAttribute", "column_attribute"),
    ("semantic_layer", "SqlAttribute", "sql_attribute"),
    ("semantic_layer", "CustomAnalysis", "custom_analysis"),
    ("semantic_layer", "PqlAnalysis", "pql_analysis"),
)


def upgrade() -> None:
    """Copy every vector onto its entity row, then remove the old schema."""
    for collection, label, destination in _BACKFILL:
        # `to_regclass` returns NULL rather than raising for a missing relation,
        # so one statement covers both the upgrade-in-place and the fresh-install
        # cases without a separate existence query.
        op.execute(
            f"""
            DO $$
            BEGIN
              IF to_regclass('vdb.{collection}') IS NOT NULL THEN
                UPDATE {destination} AS dst
                   SET embedding = src.embedding,
                       embedding_text = src.content,
                       embedding_database_name = src.database_name
                  FROM vdb.{collection} AS src
                 WHERE src.label = '{label}'
                   AND dst.id = (src.langchain_metadata ->> 'id');
              END IF;
            END $$;
            """
        )

    # CASCADE because the collections carry their own indexes and constraints.
    # Dropping the schema is what makes this migration a one-way door, and it is
    # deliberate: leaving it would preserve a second, now-unmaintained copy of
    # every vector that no code reads and nothing keeps in step.
    op.execute("DROP SCHEMA IF EXISTS vdb CASCADE")


def downgrade() -> None:
    """No-op. See the module docstring: the data lives on the entities now."""
