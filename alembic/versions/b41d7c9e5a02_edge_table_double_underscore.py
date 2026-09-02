"""Name edge tables with a double underscore

Revision ID: b41d7c9e5a02
Revises: 57abbbf6ff90
Create Date: 2026-09-02

Association tables now separate their two halves with ``__``; entity tables
never contain one. So a name tells you which kind of table it is, and where the
first entity's name ends.

Both were previously guesswork. ``column_attribute_term`` could be read as
``column_attribute`` + ``term`` or ``column`` + ``attribute_term``, and nothing
distinguished it from the ``column_attribute`` entity beside it. The same trap
sat in ``sql_attribute_term`` and ``sql_attribute_sql``.

The rule is one ``__`` per edge table, separating the first endpoint from the
rest of the name -- not ``<source>__<target>``, which cannot express the four
self-edges (``column__join``, ``table__join``) or tell apart the two edges that
share endpoints (``column__has_attribute`` and ``column__semantic_fk``).

This renames; it does not recreate. Rows, foreign keys and indexes are all
preserved, so the migration is safe on a populated database and reverses
exactly.

Constraints and indexes are renamed alongside their tables because Postgres
does not cascade a table rename to them: without this the database would end up
with ``pk_table_term`` sitting on ``table__term``, which drifts from the
metadata's naming convention and reintroduces the confusion one level down.
"""

from alembic import op

revision = "b41d7c9e5a02"
down_revision = "57abbbf6ff90"
branch_labels = None
depends_on = None


_DROP_VIEW = "DROP VIEW IF EXISTS join_path_edge"

_VIEW_BEFORE = """
CREATE OR REPLACE VIEW join_path_edge AS
    SELECT 'column'::text AS src_kind, c.id       AS src_id,
           'table'::text  AS dst_kind, c.table_id AS dst_id
      FROM catalog_column c
    UNION ALL
    SELECT 'table', c.table_id, 'column', c.id
      FROM catalog_column c
    UNION ALL
    SELECT 'column', h.column_id, 'column_attribute', h.attribute_id
      FROM column_has_attribute h
    UNION ALL
    SELECT 'column_attribute', h.attribute_id, 'column', h.column_id
      FROM column_has_attribute h
    UNION ALL
    SELECT 'column', f.column_id, 'column_attribute', f.attribute_id
      FROM column_semantic_fk f
"""

_VIEW_AFTER = """
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


def upgrade() -> None:
    op.execute(_DROP_VIEW)

    op.rename_table("column_attribute_term", "column_attribute__term")
    op.execute(
        'ALTER TABLE "column_attribute__term" RENAME CONSTRAINT "fk_column_attribute_term_attribute_id_column_attribute" TO "fk_column_attribute__term_attribute_id_column_attribute"'
    )
    op.execute(
        'ALTER TABLE "column_attribute__term" RENAME CONSTRAINT "fk_column_attribute_term_term_id_term" TO "fk_column_attribute__term_term_id_term"'
    )
    op.execute(
        'ALTER TABLE "column_attribute__term" RENAME CONSTRAINT "pk_column_attribute_term" TO "pk_column_attribute__term"'
    )
    op.execute(
        'ALTER INDEX "ix_column_attribute_term_term_id" RENAME TO "ix_column_attribute__term_term_id"'
    )

    op.rename_table("column_foreign_key", "column__foreign_key")
    op.execute(
        'ALTER TABLE "column__foreign_key" RENAME CONSTRAINT "fk_column_foreign_key_source_column_id_catalog_column" TO "fk_column__foreign_key_source_column_id_catalog_column"'
    )
    op.execute(
        'ALTER TABLE "column__foreign_key" RENAME CONSTRAINT "fk_column_foreign_key_target_column_id_catalog_column" TO "fk_column__foreign_key_target_column_id_catalog_column"'
    )
    op.execute(
        'ALTER TABLE "column__foreign_key" RENAME CONSTRAINT "pk_column_foreign_key" TO "pk_column__foreign_key"'
    )
    op.execute(
        'ALTER INDEX "ix_column_foreign_key_target" RENAME TO "ix_column__foreign_key_target"'
    )

    op.rename_table("column_has_attribute", "column__has_attribute")
    op.execute(
        'ALTER TABLE "column__has_attribute" RENAME CONSTRAINT "fk_column_has_attribute_attribute_id_column_attribute" TO "fk_column__has_attribute_attribute_id_column_attribute"'
    )
    op.execute(
        'ALTER TABLE "column__has_attribute" RENAME CONSTRAINT "fk_column_has_attribute_column_id_catalog_column" TO "fk_column__has_attribute_column_id_catalog_column"'
    )
    op.execute(
        'ALTER TABLE "column__has_attribute" RENAME CONSTRAINT "pk_column_has_attribute" TO "pk_column__has_attribute"'
    )
    op.execute(
        'ALTER INDEX "ix_column_has_attribute_attribute_id" RENAME TO "ix_column__has_attribute_attribute_id"'
    )

    op.rename_table("column_join", "column__join")
    op.execute(
        'ALTER TABLE "column__join" RENAME CONSTRAINT "fk_column_join_source_column_id_catalog_column" TO "fk_column__join_source_column_id_catalog_column"'
    )
    op.execute(
        'ALTER TABLE "column__join" RENAME CONSTRAINT "fk_column_join_target_column_id_catalog_column" TO "fk_column__join_target_column_id_catalog_column"'
    )
    op.execute(
        'ALTER TABLE "column__join" RENAME CONSTRAINT "pk_column_join" TO "pk_column__join"'
    )

    op.rename_table("column_semantic_fk", "column__semantic_fk")
    op.execute(
        'ALTER TABLE "column__semantic_fk" RENAME CONSTRAINT "fk_column_semantic_fk_attribute_id_column_attribute" TO "fk_column__semantic_fk_attribute_id_column_attribute"'
    )
    op.execute(
        'ALTER TABLE "column__semantic_fk" RENAME CONSTRAINT "fk_column_semantic_fk_column_id_catalog_column" TO "fk_column__semantic_fk_column_id_catalog_column"'
    )
    op.execute(
        'ALTER TABLE "column__semantic_fk" RENAME CONSTRAINT "pk_column_semantic_fk" TO "pk_column__semantic_fk"'
    )
    op.execute(
        'ALTER INDEX "ix_column_semantic_fk_attribute_id" RENAME TO "ix_column__semantic_fk_attribute_id"'
    )

    op.rename_table("column_union", "column__union")
    op.execute(
        'ALTER TABLE "column__union" RENAME CONSTRAINT "fk_column_union_source_column_id_catalog_column" TO "fk_column__union_source_column_id_catalog_column"'
    )
    op.execute(
        'ALTER TABLE "column__union" RENAME CONSTRAINT "fk_column_union_target_column_id_catalog_column" TO "fk_column__union_target_column_id_catalog_column"'
    )
    op.execute(
        'ALTER TABLE "column__union" RENAME CONSTRAINT "pk_column_union" TO "pk_column__union"'
    )

    op.rename_table("custom_analysis_sql", "custom_analysis__sql")
    op.execute(
        'ALTER TABLE "custom_analysis__sql" RENAME CONSTRAINT "fk_custom_analysis_sql_analysis_id_custom_analysis" TO "fk_custom_analysis__sql_analysis_id_custom_analysis"'
    )
    op.execute(
        'ALTER TABLE "custom_analysis__sql" RENAME CONSTRAINT "fk_custom_analysis_sql_sql_query_id_sql_query" TO "fk_custom_analysis__sql_sql_query_id_sql_query"'
    )
    op.execute(
        'ALTER TABLE "custom_analysis__sql" RENAME CONSTRAINT "pk_custom_analysis_sql" TO "pk_custom_analysis__sql"'
    )

    op.rename_table("sql_attribute_sql", "sql_attribute__sql")
    op.execute(
        'ALTER TABLE "sql_attribute__sql" RENAME CONSTRAINT "fk_sql_attribute_sql_attribute_id_sql_attribute" TO "fk_sql_attribute__sql_attribute_id_sql_attribute"'
    )
    op.execute(
        'ALTER TABLE "sql_attribute__sql" RENAME CONSTRAINT "fk_sql_attribute_sql_sql_query_id_sql_query" TO "fk_sql_attribute__sql_sql_query_id_sql_query"'
    )
    op.execute(
        'ALTER TABLE "sql_attribute__sql" RENAME CONSTRAINT "pk_sql_attribute_sql" TO "pk_sql_attribute__sql"'
    )

    op.rename_table("sql_attribute_term", "sql_attribute__term")
    op.execute(
        'ALTER TABLE "sql_attribute__term" RENAME CONSTRAINT "fk_sql_attribute_term_attribute_id_sql_attribute" TO "fk_sql_attribute__term_attribute_id_sql_attribute"'
    )
    op.execute(
        'ALTER TABLE "sql_attribute__term" RENAME CONSTRAINT "fk_sql_attribute_term_term_id_term" TO "fk_sql_attribute__term_term_id_term"'
    )
    op.execute(
        'ALTER TABLE "sql_attribute__term" RENAME CONSTRAINT "pk_sql_attribute_term" TO "pk_sql_attribute__term"'
    )
    op.execute(
        'ALTER INDEX "ix_sql_attribute_term_term_id" RENAME TO "ix_sql_attribute__term_term_id"'
    )

    op.rename_table("sql_query_column", "sql_query__column")
    op.execute(
        'ALTER TABLE "sql_query__column" RENAME CONSTRAINT "fk_sql_query_column_column_id_catalog_column" TO "fk_sql_query__column_column_id_catalog_column"'
    )
    op.execute(
        'ALTER TABLE "sql_query__column" RENAME CONSTRAINT "fk_sql_query_column_sql_query_id_sql_query" TO "fk_sql_query__column_sql_query_id_sql_query"'
    )
    op.execute(
        'ALTER TABLE "sql_query__column" RENAME CONSTRAINT "pk_sql_query_column" TO "pk_sql_query__column"'
    )
    op.execute(
        'ALTER INDEX "ix_sql_query_column_column_id" RENAME TO "ix_sql_query__column_column_id"'
    )

    op.rename_table("sql_query_table", "sql_query__table")
    op.execute(
        'ALTER TABLE "sql_query__table" RENAME CONSTRAINT "fk_sql_query_table_sql_query_id_sql_query" TO "fk_sql_query__table_sql_query_id_sql_query"'
    )
    op.execute(
        'ALTER TABLE "sql_query__table" RENAME CONSTRAINT "fk_sql_query_table_table_id_catalog_table" TO "fk_sql_query__table_table_id_catalog_table"'
    )
    op.execute(
        'ALTER TABLE "sql_query__table" RENAME CONSTRAINT "pk_sql_query_table" TO "pk_sql_query__table"'
    )
    op.execute(
        'ALTER INDEX "ix_sql_query_table_table_id" RENAME TO "ix_sql_query__table_table_id"'
    )

    op.rename_table("table_join", "table__join")
    op.execute(
        'ALTER TABLE "table__join" RENAME CONSTRAINT "fk_table_join_source_table_id_catalog_table" TO "fk_table__join_source_table_id_catalog_table"'
    )
    op.execute(
        'ALTER TABLE "table__join" RENAME CONSTRAINT "fk_table_join_target_table_id_catalog_table" TO "fk_table__join_target_table_id_catalog_table"'
    )
    op.execute(
        'ALTER TABLE "table__join" RENAME CONSTRAINT "pk_table_join" TO "pk_table__join"'
    )
    op.execute('ALTER INDEX "ix_table_join_target" RENAME TO "ix_table__join_target"')

    op.rename_table("table_term", "table__term")
    op.execute(
        'ALTER TABLE "table__term" RENAME CONSTRAINT "fk_table_term_table_id_catalog_table" TO "fk_table__term_table_id_catalog_table"'
    )
    op.execute(
        'ALTER TABLE "table__term" RENAME CONSTRAINT "fk_table_term_term_id_term" TO "fk_table__term_term_id_term"'
    )
    op.execute(
        'ALTER TABLE "table__term" RENAME CONSTRAINT "pk_table_term" TO "pk_table__term"'
    )
    op.execute('ALTER INDEX "ix_table_term_term_id" RENAME TO "ix_table__term_term_id"')

    op.execute(_VIEW_AFTER)


def downgrade() -> None:
    op.execute(_DROP_VIEW)

    op.execute(
        'ALTER TABLE "table__term" RENAME CONSTRAINT "fk_table__term_table_id_catalog_table" TO "fk_table_term_table_id_catalog_table"'
    )
    op.execute(
        'ALTER TABLE "table__term" RENAME CONSTRAINT "fk_table__term_term_id_term" TO "fk_table_term_term_id_term"'
    )
    op.execute(
        'ALTER TABLE "table__term" RENAME CONSTRAINT "pk_table__term" TO "pk_table_term"'
    )
    op.execute('ALTER INDEX "ix_table__term_term_id" RENAME TO "ix_table_term_term_id"')
    op.rename_table("table__term", "table_term")

    op.execute(
        'ALTER TABLE "table__join" RENAME CONSTRAINT "fk_table__join_source_table_id_catalog_table" TO "fk_table_join_source_table_id_catalog_table"'
    )
    op.execute(
        'ALTER TABLE "table__join" RENAME CONSTRAINT "fk_table__join_target_table_id_catalog_table" TO "fk_table_join_target_table_id_catalog_table"'
    )
    op.execute(
        'ALTER TABLE "table__join" RENAME CONSTRAINT "pk_table__join" TO "pk_table_join"'
    )
    op.execute('ALTER INDEX "ix_table__join_target" RENAME TO "ix_table_join_target"')
    op.rename_table("table__join", "table_join")

    op.execute(
        'ALTER TABLE "sql_query__table" RENAME CONSTRAINT "fk_sql_query__table_sql_query_id_sql_query" TO "fk_sql_query_table_sql_query_id_sql_query"'
    )
    op.execute(
        'ALTER TABLE "sql_query__table" RENAME CONSTRAINT "fk_sql_query__table_table_id_catalog_table" TO "fk_sql_query_table_table_id_catalog_table"'
    )
    op.execute(
        'ALTER TABLE "sql_query__table" RENAME CONSTRAINT "pk_sql_query__table" TO "pk_sql_query_table"'
    )
    op.execute(
        'ALTER INDEX "ix_sql_query__table_table_id" RENAME TO "ix_sql_query_table_table_id"'
    )
    op.rename_table("sql_query__table", "sql_query_table")

    op.execute(
        'ALTER TABLE "sql_query__column" RENAME CONSTRAINT "fk_sql_query__column_column_id_catalog_column" TO "fk_sql_query_column_column_id_catalog_column"'
    )
    op.execute(
        'ALTER TABLE "sql_query__column" RENAME CONSTRAINT "fk_sql_query__column_sql_query_id_sql_query" TO "fk_sql_query_column_sql_query_id_sql_query"'
    )
    op.execute(
        'ALTER TABLE "sql_query__column" RENAME CONSTRAINT "pk_sql_query__column" TO "pk_sql_query_column"'
    )
    op.execute(
        'ALTER INDEX "ix_sql_query__column_column_id" RENAME TO "ix_sql_query_column_column_id"'
    )
    op.rename_table("sql_query__column", "sql_query_column")

    op.execute(
        'ALTER TABLE "sql_attribute__term" RENAME CONSTRAINT "fk_sql_attribute__term_attribute_id_sql_attribute" TO "fk_sql_attribute_term_attribute_id_sql_attribute"'
    )
    op.execute(
        'ALTER TABLE "sql_attribute__term" RENAME CONSTRAINT "fk_sql_attribute__term_term_id_term" TO "fk_sql_attribute_term_term_id_term"'
    )
    op.execute(
        'ALTER TABLE "sql_attribute__term" RENAME CONSTRAINT "pk_sql_attribute__term" TO "pk_sql_attribute_term"'
    )
    op.execute(
        'ALTER INDEX "ix_sql_attribute__term_term_id" RENAME TO "ix_sql_attribute_term_term_id"'
    )
    op.rename_table("sql_attribute__term", "sql_attribute_term")

    op.execute(
        'ALTER TABLE "sql_attribute__sql" RENAME CONSTRAINT "fk_sql_attribute__sql_attribute_id_sql_attribute" TO "fk_sql_attribute_sql_attribute_id_sql_attribute"'
    )
    op.execute(
        'ALTER TABLE "sql_attribute__sql" RENAME CONSTRAINT "fk_sql_attribute__sql_sql_query_id_sql_query" TO "fk_sql_attribute_sql_sql_query_id_sql_query"'
    )
    op.execute(
        'ALTER TABLE "sql_attribute__sql" RENAME CONSTRAINT "pk_sql_attribute__sql" TO "pk_sql_attribute_sql"'
    )
    op.rename_table("sql_attribute__sql", "sql_attribute_sql")

    op.execute(
        'ALTER TABLE "custom_analysis__sql" RENAME CONSTRAINT "fk_custom_analysis__sql_analysis_id_custom_analysis" TO "fk_custom_analysis_sql_analysis_id_custom_analysis"'
    )
    op.execute(
        'ALTER TABLE "custom_analysis__sql" RENAME CONSTRAINT "fk_custom_analysis__sql_sql_query_id_sql_query" TO "fk_custom_analysis_sql_sql_query_id_sql_query"'
    )
    op.execute(
        'ALTER TABLE "custom_analysis__sql" RENAME CONSTRAINT "pk_custom_analysis__sql" TO "pk_custom_analysis_sql"'
    )
    op.rename_table("custom_analysis__sql", "custom_analysis_sql")

    op.execute(
        'ALTER TABLE "column__union" RENAME CONSTRAINT "fk_column__union_source_column_id_catalog_column" TO "fk_column_union_source_column_id_catalog_column"'
    )
    op.execute(
        'ALTER TABLE "column__union" RENAME CONSTRAINT "fk_column__union_target_column_id_catalog_column" TO "fk_column_union_target_column_id_catalog_column"'
    )
    op.execute(
        'ALTER TABLE "column__union" RENAME CONSTRAINT "pk_column__union" TO "pk_column_union"'
    )
    op.rename_table("column__union", "column_union")

    op.execute(
        'ALTER TABLE "column__semantic_fk" RENAME CONSTRAINT "fk_column__semantic_fk_attribute_id_column_attribute" TO "fk_column_semantic_fk_attribute_id_column_attribute"'
    )
    op.execute(
        'ALTER TABLE "column__semantic_fk" RENAME CONSTRAINT "fk_column__semantic_fk_column_id_catalog_column" TO "fk_column_semantic_fk_column_id_catalog_column"'
    )
    op.execute(
        'ALTER TABLE "column__semantic_fk" RENAME CONSTRAINT "pk_column__semantic_fk" TO "pk_column_semantic_fk"'
    )
    op.execute(
        'ALTER INDEX "ix_column__semantic_fk_attribute_id" RENAME TO "ix_column_semantic_fk_attribute_id"'
    )
    op.rename_table("column__semantic_fk", "column_semantic_fk")

    op.execute(
        'ALTER TABLE "column__join" RENAME CONSTRAINT "fk_column__join_source_column_id_catalog_column" TO "fk_column_join_source_column_id_catalog_column"'
    )
    op.execute(
        'ALTER TABLE "column__join" RENAME CONSTRAINT "fk_column__join_target_column_id_catalog_column" TO "fk_column_join_target_column_id_catalog_column"'
    )
    op.execute(
        'ALTER TABLE "column__join" RENAME CONSTRAINT "pk_column__join" TO "pk_column_join"'
    )
    op.rename_table("column__join", "column_join")

    op.execute(
        'ALTER TABLE "column__has_attribute" RENAME CONSTRAINT "fk_column__has_attribute_attribute_id_column_attribute" TO "fk_column_has_attribute_attribute_id_column_attribute"'
    )
    op.execute(
        'ALTER TABLE "column__has_attribute" RENAME CONSTRAINT "fk_column__has_attribute_column_id_catalog_column" TO "fk_column_has_attribute_column_id_catalog_column"'
    )
    op.execute(
        'ALTER TABLE "column__has_attribute" RENAME CONSTRAINT "pk_column__has_attribute" TO "pk_column_has_attribute"'
    )
    op.execute(
        'ALTER INDEX "ix_column__has_attribute_attribute_id" RENAME TO "ix_column_has_attribute_attribute_id"'
    )
    op.rename_table("column__has_attribute", "column_has_attribute")

    op.execute(
        'ALTER TABLE "column__foreign_key" RENAME CONSTRAINT "fk_column__foreign_key_source_column_id_catalog_column" TO "fk_column_foreign_key_source_column_id_catalog_column"'
    )
    op.execute(
        'ALTER TABLE "column__foreign_key" RENAME CONSTRAINT "fk_column__foreign_key_target_column_id_catalog_column" TO "fk_column_foreign_key_target_column_id_catalog_column"'
    )
    op.execute(
        'ALTER TABLE "column__foreign_key" RENAME CONSTRAINT "pk_column__foreign_key" TO "pk_column_foreign_key"'
    )
    op.execute(
        'ALTER INDEX "ix_column__foreign_key_target" RENAME TO "ix_column_foreign_key_target"'
    )
    op.rename_table("column__foreign_key", "column_foreign_key")

    op.execute(
        'ALTER TABLE "column_attribute__term" RENAME CONSTRAINT "fk_column_attribute__term_attribute_id_column_attribute" TO "fk_column_attribute_term_attribute_id_column_attribute"'
    )
    op.execute(
        'ALTER TABLE "column_attribute__term" RENAME CONSTRAINT "fk_column_attribute__term_term_id_term" TO "fk_column_attribute_term_term_id_term"'
    )
    op.execute(
        'ALTER TABLE "column_attribute__term" RENAME CONSTRAINT "pk_column_attribute__term" TO "pk_column_attribute_term"'
    )
    op.execute(
        'ALTER INDEX "ix_column_attribute__term_term_id" RENAME TO "ix_column_attribute_term_term_id"'
    )
    op.rename_table("column_attribute__term", "column_attribute_term")

    op.execute(_VIEW_BEFORE)
