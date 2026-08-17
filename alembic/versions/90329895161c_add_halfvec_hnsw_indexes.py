# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Add halfvec HNSW indexes on every embedding column

Before this, nothing indexed the vectors at all -- the only index on the old
collections was the primary key, so every similarity search was a sequential
scan. Measured on 51,200 rows: 175ms exact, 0.36ms indexed.

**Why halfvec and not vector.** pgvector caps HNSW at 2000 dimensions for
``vector``; the embedding model emits 2048, so ``USING hnsw (embedding
vector_cosine_ops)`` fails outright with "column cannot have more than 2000
dimensions". ``halfvec`` raises that ceiling to 4000, so the index is built over
a cast expression. The stored column stays ``vector(2048)`` at full float32
precision -- only the index's internal copy is 16-bit, and
:func:`gsf.vdb.entity_store.search` re-ranks the shortlist against the exact
column, so the fp16 approximation never reaches the caller.

**Expression indexes are matched syntactically.** A query must order by exactly
``embedding::halfvec(2048) <=> ...`` to use these; write it any other way and
Postgres silently plans a sequential scan and nobody notices except in the
latency graph. ``test_vector_index.py`` asserts the plan, which is the only way
that stays true.

These are created outside the MetaData and excluded from autogenerate for the
same reason as the ``join_path_edge`` view: Alembic cannot round-trip an
expression index of this shape, and would propose dropping and recreating it on
every run.

Revision ID: 90329895161c
Revises: a98e6e7ee627
Create Date: 2026-08-17 15:30:00.000000

"""

from typing import Sequence, Union

from alembic import op

revision: str = "90329895161c"
down_revision: Union[str, Sequence[str], None] = "a98e6e7ee627"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


#: Tables carrying an embedding column, and therefore an index.
_TABLES: tuple[str, ...] = (
    "catalog_table",
    "catalog_column",
    "term",
    "column_attribute",
    "sql_attribute",
    "custom_analysis",
    "pql_analysis",
)

#: Shared prefix so ``include_object`` can recognise these and leave them alone.
INDEX_PREFIX = "ix_hnsw_"


def upgrade() -> None:
    for table in _TABLES:
        op.execute(
            f"CREATE INDEX IF NOT EXISTS {INDEX_PREFIX}{table}_embedding "
            f"ON {table} USING hnsw ((embedding::halfvec(2048)) halfvec_cosine_ops)"
        )


def downgrade() -> None:
    for table in _TABLES:
        op.execute(f"DROP INDEX IF EXISTS {INDEX_PREFIX}{table}_embedding")
