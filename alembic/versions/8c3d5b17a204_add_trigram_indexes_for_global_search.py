# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Add trigram indexes for global search

Global search matches a *substring*: typing ``mount`` has to find
``total_amount``, and the API offers no other mode -- ``contains`` is the only
``text_match_option`` the service accepts.

Nothing already in the schema can serve that. A btree needs a left anchor, so it
cannot answer an unanchored ``LIKE`` at all, and ``tsvector`` reaches only the
front of a token: ``to_tsquery('mount:*')`` does not match the word it sits
inside of. Without a trigram index every keystroke is a sequential scan of nine
tables, twice each -- once for ``name`` and once for ``description``.

The operator class folds case itself, so the indexed expression is the bare
column rather than ``lower(name)``. That is deliberate and load-bearing in one
direction: ``ILIKE`` matches these as written, and wrapping them in ``lower()``
would build indexes that the queries in ``auto_ontology.dal.search`` could no longer use.

The table list is inlined rather than imported from ``auto_ontology.dal.schema``. Importing
it would make this revision create whatever the application currently defines
instead of what it defined here, so adding a searchable entity later would
silently change what this migration does. A migration is a record of what ran.
"""

from typing import Sequence, Union

from alembic import op

# Kept in step with `TRIGRAM_SEARCH_TABLES` in `auto_ontology/dal/schema.py`, which
# declares the same indexes to the metadata so autogenerate does not propose
# dropping them on the next run.
SEARCH_TABLES = (
    "catalog_database",
    "catalog_schema",
    "catalog_table",
    "catalog_column",
    "custom_analysis",
    "term",
    "column_attribute",
    "sql_attribute",
    "pql_analysis",
)

SEARCH_COLUMNS = ("name", "description")

# revision identifiers, used by Alembic.
revision: str = "8c3d5b17a204"
down_revision: Union[str, Sequence[str], None] = "57abbbf6ff90"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the extension, then one GIN index per searchable column."""
    # Unqualified, like every other statement here: `search_path` is pinned to
    # `public` on the migration connection. `pg_trgm` ships with the standard
    # contrib set and is present in the `pgvector/pgvector` images the compose
    # file and the chart use, but creating it needs privileges a locked-down
    # managed instance may withhold -- in which case this migration is the right
    # place to fail, loudly, rather than the first search of the day.
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    for table in SEARCH_TABLES:
        for column in SEARCH_COLUMNS:
            op.create_index(
                f"ix_{table}_{column}_trgm",
                table,
                [column],
                unique=False,
                postgresql_using="gin",
                postgresql_ops={column: "gin_trgm_ops"},
            )


def downgrade() -> None:
    """Drop the indexes, but leave the extension.

    ``DROP EXTENSION`` is not the inverse of ``CREATE EXTENSION IF NOT EXISTS``:
    the extension may have predated this migration, and anything else in the
    database using a trigram index would break with it. Dropping what this
    revision definitely created is the reversible part.
    """
    for table in reversed(SEARCH_TABLES):
        for column in reversed(SEARCH_COLUMNS):
            op.drop_index(f"ix_{table}_{column}_trgm", table_name=table)
