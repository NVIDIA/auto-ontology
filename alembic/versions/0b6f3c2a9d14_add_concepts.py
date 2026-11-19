# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""add concepts

Revision ID: 0b6f3c2a9d14
Revises: caee572fa81c
Create Date: 2026-11-18 12:00:00.000000

Every Term that represents a table gets a one-member Concept, and every
semantic FK is re-pointed from the ColumnAttribute it targeted to a key on that
attribute's Concept, held by the attribute's own column. Joins after the
upgrade therefore land on exactly the columns they landed on before it.

The unused ``join_path_edge`` view is dropped: it read
``column__semantic_fk.attribute_id``, which this revision removes.
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0b6f3c2a9d14"
down_revision: str | Sequence[str] | None = "caee572fa81c"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_UUID = sa.text("(gen_random_uuid())::text")

_JOIN_PATH_EDGE_VIEW_SQL = """
CREATE OR REPLACE VIEW join_path_edge AS
    SELECT 'column'::text AS src_kind, c.id       AS src_id,
           'table'::text  AS dst_kind, c.table_id AS dst_id
      FROM catalog_column c
    UNION ALL
    SELECT 'table', c.table_id, 'column', c.id
      FROM catalog_column c
    UNION ALL
    SELECT 'column', h.column_id, 'column_attribute', h.attribute_id
      FROM column__has_attribute h
    UNION ALL
    SELECT 'column_attribute', h.attribute_id, 'column', h.column_id
      FROM column__has_attribute h
    UNION ALL
    SELECT 'column', f.column_id, 'column_attribute', f.attribute_id
      FROM column__semantic_fk f
"""


def _create_tables() -> None:
    op.create_table(
        "concept",
        sa.Column("id", sa.Text(), server_default=_UUID, nullable=False),
        sa.Column("database_id", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "source", sa.Text(), server_default=sa.text("'semantic'"), nullable=False
        ),
        sa.Column(
            "certified", sa.Boolean(), server_default=sa.text("false"), nullable=False
        ),
        sa.Column(
            "created",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.Column("modified", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_by", sa.Text(), nullable=True),
        sa.Column("modified_by", sa.Text(), nullable=True),
        sa.Column("imported_id", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["database_id"],
            ["catalog_database.id"],
            name=op.f("fk_concept_database_id_catalog_database"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_concept")),
        sa.UniqueConstraint("database_id", "name", name="uq_concept_database_name"),
    )
    op.create_index("ix_concept_database_id", "concept", ["database_id"])
    op.create_index("ix_concept_imported_id", "concept", ["imported_id"])
    op.create_table(
        "term__concept",
        sa.Column("term_id", sa.Text(), nullable=False),
        sa.Column("concept_id", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["concept_id"],
            ["concept.id"],
            name=op.f("fk_term__concept_concept_id_concept"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["term_id"],
            ["term.id"],
            name=op.f("fk_term__concept_term_id_term"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("term_id", "concept_id", name=op.f("pk_term__concept")),
        sa.UniqueConstraint("term_id", name=op.f("uq_term__concept_term_id")),
    )
    op.create_index("ix_term__concept_concept_id", "term__concept", ["concept_id"])
    op.create_table(
        "concept_key",
        sa.Column("id", sa.Text(), server_default=_UUID, nullable=False),
        sa.Column("concept_id", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["concept_id"],
            ["concept.id"],
            name=op.f("fk_concept_key_concept_id_concept"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_concept_key")),
    )
    op.create_index("ix_concept_key_concept_id", "concept_key", ["concept_id"])
    op.create_table(
        "concept_key__column",
        sa.Column("key_id", sa.Text(), nullable=False),
        sa.Column("column_id", sa.Text(), nullable=False),
        sa.Column("table_id", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["column_id"],
            ["catalog_column.id"],
            name=op.f("fk_concept_key__column_column_id_catalog_column"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["key_id"],
            ["concept_key.id"],
            name=op.f("fk_concept_key__column_key_id_concept_key"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["table_id"],
            ["catalog_table.id"],
            name=op.f("fk_concept_key__column_table_id_catalog_table"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "key_id", "column_id", name=op.f("pk_concept_key__column")
        ),
        sa.UniqueConstraint("column_id", name=op.f("uq_concept_key__column_column_id")),
        sa.UniqueConstraint(
            "key_id", "table_id", name=op.f("uq_concept_key__column_key_id")
        ),
    )
    op.create_index(
        "ix_concept_key__column_table_id", "concept_key__column", ["table_id"]
    )


def _backfill_concepts() -> None:
    # A Term's database is that of the first table it represents; a Term that
    # represents none has nothing for a Concept to be scoped to. The Concept
    # takes the Term's id, which nothing else needs to map back from. A name
    # already taken in the database is qualified the way terms.py qualifies.
    op.execute(
        """
        WITH term_db AS (
            SELECT DISTINCT ON (tt.term_id)
                   tt.term_id, s.database_id
              FROM table__term tt
              JOIN catalog_table t ON t.id = tt.table_id
              JOIN catalog_schema s ON s.id = t.schema_id
             ORDER BY tt.term_id, tt.table_id
        ), named AS (
            SELECT term.id, term_db.database_id, term.name, term.description,
                   row_number() OVER (
                       PARTITION BY term_db.database_id, term.name
                       ORDER BY term.id
                   ) AS rank
              FROM term JOIN term_db ON term_db.term_id = term.id
        )
        INSERT INTO concept (id, database_id, name, description)
        SELECT id, database_id,
               CASE WHEN rank = 1 THEN name ELSE name || ' (' || id || ')' END,
               description
          FROM named
        """
    )
    op.execute(
        "INSERT INTO term__concept (term_id, concept_id) SELECT id, id FROM concept"
    )


def _backfill_keys() -> None:
    # One key per column that a semantic FK lands on today, plus one per
    # single-column declared primary key that none lands on.
    op.execute(
        """
        CREATE TEMPORARY TABLE _concept_key_column ON COMMIT DROP AS
        WITH targets AS (
            SELECT DISTINCT ON (h.attribute_id) h.attribute_id, h.column_id
              FROM column__has_attribute h
              JOIN (SELECT DISTINCT attribute_id FROM column__semantic_fk) f
                ON f.attribute_id = h.attribute_id
             ORDER BY h.attribute_id, h.column_id
        ), key_columns AS (
            SELECT column_id FROM targets
            UNION
            SELECT c.id
              FROM catalog_table t
              JOIN catalog_column c ON c.table_id = t.id
             WHERE cardinality(t.pk) = 1 AND c.name = t.pk[1]
        )
        SELECT (gen_random_uuid())::text AS key_id, c.id AS column_id,
               c.table_id, c.name, tc.concept_id
          FROM key_columns k
          JOIN catalog_column c ON c.id = k.column_id
          JOIN table__term tt ON tt.table_id = c.table_id
          JOIN term__concept tc ON tc.term_id = tt.term_id
        """
    )
    # A table represented by several Terms would list its column once per
    # Term; keep one, and only one key column per table and Concept.
    op.execute(
        """
        DELETE FROM _concept_key_column k
         USING _concept_key_column other
         WHERE k.column_id = other.column_id AND k.key_id > other.key_id
        """
    )
    op.execute(
        """
        INSERT INTO concept_key (id, concept_id, name)
        SELECT key_id, concept_id, name FROM _concept_key_column
        """
    )
    op.execute(
        """
        INSERT INTO concept_key__column (key_id, column_id, table_id)
        SELECT key_id, column_id, table_id FROM _concept_key_column
        """
    )


def _repoint_semantic_fks() -> None:
    op.add_column("column__semantic_fk", sa.Column("key_id", sa.Text(), nullable=True))
    op.execute(
        """
        UPDATE column__semantic_fk f
           SET key_id = kc.key_id
          FROM (
              SELECT DISTINCT ON (h.attribute_id) h.attribute_id, h.column_id
                FROM column__has_attribute h
               ORDER BY h.attribute_id, h.column_id
          ) owner
          JOIN concept_key__column kc ON kc.column_id = owner.column_id
         WHERE owner.attribute_id = f.attribute_id
        """
    )
    # A column references one key from here on. Of several edges on one column,
    # keep the one joins used, then the lowest attribute id.
    op.execute(
        """
        DELETE FROM column__semantic_fk f
         WHERE f.key_id IS NULL
            OR EXISTS (
                SELECT 1 FROM column__semantic_fk other
                 WHERE other.column_id = f.column_id
                   AND other.key_id IS NOT NULL
                   AND (other.use_in_join_paths, f.attribute_id)
                       > (f.use_in_join_paths, other.attribute_id)
            )
        """
    )
    op.drop_index("ix_column__semantic_fk_attribute_id", "column__semantic_fk")
    op.drop_constraint(
        "fk_column__semantic_fk_attribute_id_column_attribute",
        "column__semantic_fk",
        type_="foreignkey",
    )
    op.drop_constraint("pk_column__semantic_fk", "column__semantic_fk", type_="primary")
    op.drop_column("column__semantic_fk", "attribute_id")
    op.alter_column("column__semantic_fk", "key_id", nullable=False)
    op.create_primary_key(
        "pk_column__semantic_fk", "column__semantic_fk", ["column_id"]
    )
    op.create_foreign_key(
        op.f("fk_column__semantic_fk_key_id_concept_key"),
        "column__semantic_fk",
        "concept_key",
        ["key_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index("ix_column__semantic_fk_key_id", "column__semantic_fk", ["key_id"])


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("DROP VIEW IF EXISTS join_path_edge")
    _create_tables()
    _backfill_concepts()
    _backfill_keys()
    _repoint_semantic_fks()


def downgrade() -> None:
    """Downgrade schema."""
    op.add_column(
        "column__semantic_fk", sa.Column("attribute_id", sa.Text(), nullable=True)
    )
    # Back to an attribute: the lowest one held by any of the key's columns.
    op.execute(
        """
        UPDATE column__semantic_fk f
           SET attribute_id = target.attribute_id
          FROM (
              SELECT kc.key_id, min(h.attribute_id) AS attribute_id
                FROM concept_key__column kc
                JOIN column__has_attribute h ON h.column_id = kc.column_id
               GROUP BY kc.key_id
          ) target
         WHERE target.key_id = f.key_id
        """
    )
    op.execute("DELETE FROM column__semantic_fk WHERE attribute_id IS NULL")
    op.drop_index("ix_column__semantic_fk_key_id", "column__semantic_fk")
    op.drop_constraint(
        "fk_column__semantic_fk_key_id_concept_key",
        "column__semantic_fk",
        type_="foreignkey",
    )
    op.drop_constraint("pk_column__semantic_fk", "column__semantic_fk", type_="primary")
    op.drop_column("column__semantic_fk", "key_id")
    op.alter_column("column__semantic_fk", "attribute_id", nullable=False)
    op.create_primary_key(
        "pk_column__semantic_fk", "column__semantic_fk", ["column_id", "attribute_id"]
    )
    op.create_foreign_key(
        "fk_column__semantic_fk_attribute_id_column_attribute",
        "column__semantic_fk",
        "column_attribute",
        ["attribute_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index(
        "ix_column__semantic_fk_attribute_id", "column__semantic_fk", ["attribute_id"]
    )
    op.drop_index("ix_concept_key__column_table_id", "concept_key__column")
    op.drop_table("concept_key__column")
    op.drop_index("ix_concept_key_concept_id", "concept_key")
    op.drop_table("concept_key")
    op.drop_index("ix_term__concept_concept_id", "term__concept")
    op.drop_table("term__concept")
    op.drop_index("ix_concept_imported_id", "concept")
    op.drop_index("ix_concept_database_id", "concept")
    op.drop_table("concept")
    op.execute(_JOIN_PATH_EDGE_VIEW_SQL)
