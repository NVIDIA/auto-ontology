# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""add concepts and move semantic fks onto them

Every Term gets its own Concept. A concept's key on each of the term's tables
is the column existing semantic FKs point at there, or the table's declared
primary key when nothing points at it. Each ``column__semantic_fk`` row then
becomes a one-column ``concept__fk`` reference, and the old table is dropped.

Revision ID: 1c68f8419561
Revises: 08de5bc4b3c3
Create Date: 2026-11-06 21:43:44.112223

"""

from collections import Counter, defaultdict
from collections.abc import Sequence
import logging

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "1c68f8419561"
down_revision: str | Sequence[str] | None = "08de5bc4b3c3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

logger = logging.getLogger("alembic.runtime.migration")

_VIEW_HEAD = """
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
"""
_VIEW_SEMANTIC_FK_LEG = """
    UNION ALL
    SELECT 'column', f.column_id, 'column_attribute', f.attribute_id
      FROM column__semantic_fk f
"""


def _create_concept_tables() -> None:
    op.create_table(
        "concept",
        sa.Column(
            "id",
            sa.Text(),
            server_default=sa.text("(gen_random_uuid())::text"),
            nullable=False,
        ),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "source", sa.Text(), server_default=sa.text("'semantic'"), nullable=False
        ),
        sa.Column(
            "name_certified",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column(
            "description_certified",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column("owner_id", sa.Text(), nullable=True),
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
        sa.PrimaryKeyConstraint("id", name=op.f("pk_concept")),
        sa.UniqueConstraint("name", "source", name="uq_concept_name_source"),
    )
    op.create_index("ix_concept_imported_id", "concept", ["imported_id"], unique=False)
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
    op.create_index(
        "ix_term__concept_concept_id", "term__concept", ["concept_id"], unique=False
    )
    op.create_table(
        "concept__fk",
        sa.Column(
            "id",
            sa.Text(),
            server_default=sa.text("(gen_random_uuid())::text"),
            nullable=False,
        ),
        sa.Column("concept_id", sa.Text(), nullable=False),
        sa.Column("source_table_id", sa.Text(), nullable=False),
        sa.Column(
            "use_in_join_paths",
            sa.Boolean(),
            server_default=sa.text("true"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["concept_id"],
            ["concept.id"],
            name=op.f("fk_concept__fk_concept_id_concept"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["source_table_id"],
            ["catalog_table.id"],
            name=op.f("fk_concept__fk_source_table_id_catalog_table"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_concept__fk")),
    )
    op.create_index(
        "ix_concept__fk_concept_id", "concept__fk", ["concept_id"], unique=False
    )
    op.create_index(
        "ix_concept__fk_source_table_id",
        "concept__fk",
        ["source_table_id"],
        unique=False,
    )
    op.create_table(
        "concept__fk_column",
        sa.Column("fk_id", sa.Text(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("column_id", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["column_id"],
            ["catalog_column.id"],
            name=op.f("fk_concept__fk_column_column_id_catalog_column"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["fk_id"],
            ["concept__fk.id"],
            name=op.f("fk_concept__fk_column_fk_id_concept__fk"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("fk_id", "ordinal", name=op.f("pk_concept__fk_column")),
        sa.UniqueConstraint("column_id", name=op.f("uq_concept__fk_column_column_id")),
    )
    op.create_table(
        "concept__pk",
        sa.Column("concept_id", sa.Text(), nullable=False),
        sa.Column("column_id", sa.Text(), nullable=False),
        sa.Column("table_id", sa.Text(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["column_id"],
            ["catalog_column.id"],
            name=op.f("fk_concept__pk_column_id_catalog_column"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["concept_id"],
            ["concept.id"],
            name=op.f("fk_concept__pk_concept_id_concept"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["table_id"],
            ["catalog_table.id"],
            name=op.f("fk_concept__pk_table_id_catalog_table"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("concept_id", "column_id", name=op.f("pk_concept__pk")),
        sa.UniqueConstraint(
            "concept_id", "table_id", "ordinal", name="uq_concept__pk_table_ordinal"
        ),
    )
    op.create_index("ix_concept__pk_column_id", "concept__pk", ["column_id"], unique=False)
    op.create_index("ix_concept__pk_table_id", "concept__pk", ["table_id"], unique=False)


def _backfill() -> None:
    bind = op.get_bind()
    bind.execute(
        sa.text(
            """
            INSERT INTO concept (name, description, source, name_certified,
                                 description_certified, owner_id, imported_id)
            SELECT name, description, source, name_certified,
                   description_certified, owner_id, imported_id
              FROM term
            """
        )
    )
    bind.execute(
        sa.text(
            """
            INSERT INTO term__concept (term_id, concept_id)
            SELECT t.id, c.id
              FROM term t
              JOIN concept c ON c.name = t.name AND c.source = t.source
            """
        )
    )
    concept_of_term = {
        r.term_id: r.concept_id
        for r in bind.execute(sa.text("SELECT term_id, concept_id FROM term__concept"))
    }

    # Each semantic FK: source column, its table, the referenced column and
    # table, and the referenced attribute's term.
    fk_rows = bind.execute(
        sa.text(
            """
            SELECT f.column_id, src.table_id AS source_table_id,
                   f.use_in_join_paths, owner.id AS target_column_id,
                   owner.table_id AS target_table_id, cat.term_id
              FROM column__semantic_fk f
              JOIN catalog_column src ON src.id = f.column_id
              JOIN column__has_attribute h ON h.attribute_id = f.attribute_id
              JOIN catalog_column owner ON owner.id = h.column_id
              JOIN column_attribute__term cat ON cat.attribute_id = f.attribute_id
             ORDER BY f.column_id, owner.id
            """
        )
    ).fetchall()

    targeted: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for r in fk_rows:
        concept_id = concept_of_term.get(r.term_id)
        if concept_id:
            targeted[(concept_id, r.target_table_id)][r.target_column_id] += 1

    tables = bind.execute(
        sa.text(
            """
            SELECT tt.term_id, tb.id AS table_id, tb.pk
              FROM table__term tt
              JOIN catalog_table tb ON tb.id = tt.table_id
            """
        )
    ).fetchall()
    columns_by_table: dict[str, dict[str, str]] = defaultdict(dict)
    for r in bind.execute(sa.text("SELECT id, table_id, name FROM catalog_column")):
        columns_by_table[r.table_id][r.name] = r.id

    keys: dict[str, dict[str, list[str]]] = defaultdict(dict)
    for r in tables:
        concept_id = concept_of_term.get(r.term_id)
        if not concept_id:
            continue
        hits = targeted.get((concept_id, r.table_id))
        names = columns_by_table.get(r.table_id, {})
        declared = [names[n] for n in (r.pk or []) if n in names]
        if hits:
            # The most-referenced column; a declared PK column wins a tie.
            key = [
                min(
                    hits,
                    key=lambda cid: (-hits[cid], cid not in declared, cid),
                )
            ]
        elif declared and len(declared) == len(r.pk or []):
            key = declared
        else:
            continue
        keys[concept_id][r.table_id] = key

    pk_rows = []
    for concept_id, per_table in keys.items():
        arity = Counter(len(k) for k in per_table.values()).most_common(1)[0][0]
        for table_id, key in sorted(per_table.items()):
            if len(key) != arity:
                logger.warning(
                    "concept %s: key on table %s has %d column(s), expected %d; "
                    "left out",
                    concept_id,
                    table_id,
                    len(key),
                    arity,
                )
                continue
            pk_rows.extend(
                {
                    "concept_id": concept_id,
                    "column_id": column_id,
                    "table_id": table_id,
                    "ordinal": ordinal,
                }
                for ordinal, column_id in enumerate(key)
            )
    if pk_rows:
        bind.execute(
            sa.text(
                "INSERT INTO concept__pk (concept_id, column_id, table_id, ordinal) "
                "VALUES (:concept_id, :column_id, :table_id, :ordinal) "
                "ON CONFLICT DO NOTHING"
            ),
            pk_rows,
        )
    single_key_columns = {
        (row["concept_id"], row["column_id"])
        for row in pk_rows
        if len(keys[row["concept_id"]][row["table_id"]]) == 1
    }

    linked: set[str] = set()
    skipped = 0
    for r in fk_rows:
        concept_id = concept_of_term.get(r.term_id)
        if r.column_id in linked:
            continue
        if (concept_id, r.target_column_id) not in single_key_columns:
            skipped += 1
            continue
        fk_id = bind.execute(
            sa.text(
                "INSERT INTO concept__fk (concept_id, source_table_id, "
                "use_in_join_paths) VALUES (:c, :t, :u) RETURNING id"
            ),
            {"c": concept_id, "t": r.source_table_id, "u": r.use_in_join_paths},
        ).scalar_one()
        bind.execute(
            sa.text(
                "INSERT INTO concept__fk_column (fk_id, ordinal, column_id) "
                "VALUES (:f, 0, :col)"
            ),
            {"f": fk_id, "col": r.column_id},
        )
        linked.add(r.column_id)
    logger.info(
        "Moved %d semantic FK(s) onto concepts; %d not keyed on their concept "
        "were left out",
        len(linked),
        skipped,
    )


def upgrade() -> None:
    """Upgrade schema."""
    _create_concept_tables()
    _backfill()
    op.execute(_VIEW_HEAD)
    op.drop_index(
        op.f("ix_column__semantic_fk_attribute_id"), table_name="column__semantic_fk"
    )
    op.drop_table("column__semantic_fk")


def downgrade() -> None:
    """Downgrade schema."""
    op.create_table(
        "column__semantic_fk",
        sa.Column("column_id", sa.TEXT(), autoincrement=False, nullable=False),
        sa.Column("attribute_id", sa.TEXT(), autoincrement=False, nullable=False),
        sa.Column(
            "use_in_join_paths",
            sa.BOOLEAN(),
            server_default=sa.text("true"),
            autoincrement=False,
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["attribute_id"],
            ["column_attribute.id"],
            name=op.f("fk_column__semantic_fk_attribute_id_column_attribute"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["column_id"],
            ["catalog_column.id"],
            name=op.f("fk_column__semantic_fk_column_id_catalog_column"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "column_id", "attribute_id", name=op.f("pk_column__semantic_fk")
        ),
    )
    op.create_index(
        op.f("ix_column__semantic_fk_attribute_id"),
        "column__semantic_fk",
        ["attribute_id"],
        unique=False,
    )
    # A one-column reference points at the attribute on its concept's key column
    # (the first member table's, by id). Composite references have no old form.
    op.execute(
        """
        INSERT INTO column__semantic_fk (column_id, attribute_id, use_in_join_paths)
        SELECT DISTINCT ON (fc.column_id) fc.column_id, h.attribute_id,
               f.use_in_join_paths
          FROM concept__fk f
          JOIN concept__fk_column fc ON fc.fk_id = f.id
          JOIN concept__pk pk ON pk.concept_id = f.concept_id AND pk.ordinal = 0
          JOIN column__has_attribute h ON h.column_id = pk.column_id
         WHERE NOT EXISTS (
                 SELECT 1 FROM concept__fk_column other
                  WHERE other.fk_id = f.id AND other.ordinal > 0)
         ORDER BY fc.column_id, pk.table_id, h.attribute_id
        """
    )
    op.execute(_VIEW_HEAD + _VIEW_SEMANTIC_FK_LEG)
    op.drop_index("ix_concept__pk_table_id", table_name="concept__pk")
    op.drop_index("ix_concept__pk_column_id", table_name="concept__pk")
    op.drop_table("concept__pk")
    op.drop_table("concept__fk_column")
    op.drop_index("ix_concept__fk_source_table_id", table_name="concept__fk")
    op.drop_index("ix_concept__fk_concept_id", table_name="concept__fk")
    op.drop_table("concept__fk")
    op.drop_index("ix_term__concept_concept_id", table_name="term__concept")
    op.drop_table("term__concept")
    op.drop_index("ix_concept_imported_id", table_name="concept")
    op.drop_table("concept")
