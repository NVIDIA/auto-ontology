"""Add junction-table classification to catalog tables.

Revision ID: 2c24b6498f41
Revises: e37f4a1f2f5d
Create Date: 2026-09-14 13:45:45.631948

The backfill preserves bridge classifications created before the catalog flag
existed.  A bridge SqlAttribute's parsed SQL already links to every referenced
catalog table through ``sql_query__table``.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "2c24b6498f41"
down_revision: Union[str, Sequence[str], None] = "e37f4a1f2f5d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add the flag and recover classifications from existing bridge SQL."""
    op.add_column(
        "catalog_table",
        sa.Column(
            "is_junction_table",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
    )
    op.execute(
        """
        WITH resolved_fk_columns AS (
            SELECT foreign_key.source_column_id AS column_id
            FROM column__foreign_key AS foreign_key
            JOIN catalog_column AS target
              ON target.id = foreign_key.target_column_id

            UNION

            SELECT semantic_fk.column_id
            FROM column__semantic_fk AS semantic_fk
            JOIN column__has_attribute AS owner
              ON owner.attribute_id = semantic_fk.attribute_id
            JOIN catalog_column AS target
              ON target.id = owner.column_id
        ),
        bridge_referenced_tables AS (
            SELECT DISTINCT query_table.table_id
            FROM sql_attribute AS attribute
            JOIN sql_attribute__sql AS attribute_sql
              ON attribute_sql.attribute_id = attribute.id
            JOIN sql_query__table AS query_table
              ON query_table.sql_query_id = attribute_sql.sql_query_id
            WHERE attribute.source = 'bridgeTable'
        )
        UPDATE catalog_table AS catalog
        SET is_junction_table = true
        FROM bridge_referenced_tables AS bridge
        WHERE bridge.table_id = catalog.id
          AND 2 <= (
              SELECT count(*)
              FROM catalog_column AS candidate_column
              WHERE candidate_column.table_id = catalog.id
          )
          AND NOT EXISTS (
              SELECT 1
              FROM catalog_column AS candidate_column
              WHERE candidate_column.table_id = catalog.id
                AND NOT EXISTS (
                    SELECT 1
                    FROM resolved_fk_columns AS resolved
                    WHERE resolved.column_id = candidate_column.id
                )
          )
          AND NOT EXISTS (
              SELECT 1
              FROM catalog_column AS candidate_column
              JOIN column__has_attribute AS owned_attribute
                ON owned_attribute.column_id = candidate_column.id
              WHERE candidate_column.table_id = catalog.id
          )
        """
    )


def downgrade() -> None:
    """Remove the junction-table flag."""
    op.drop_column("catalog_table", "is_junction_table")
