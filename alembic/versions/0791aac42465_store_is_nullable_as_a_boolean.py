"""store is_nullable as a boolean

Revision ID: 0791aac42465
Revises: 63f6c85ccfce
Create Date: 2026-08-12 14:17:57.498135

``catalog_column.is_nullable`` held the *strings* ``information_schema``
reports, ``'YES'`` and ``'NO'``. Both are truthy, so every reader that did the
obvious thing decided the entire catalog was nullable.

Autogenerate emitted a bare ``ALTER COLUMN ... TYPE boolean``, which Postgres
refuses on a populated table ("cannot be cast automatically"). The ``USING``
clauses below do the conversion explicitly. Postgres would in fact accept
``is_nullable::boolean`` — ``'YES'``/``'NO'`` are valid boolean input literals —
but spelling out the comparison keeps the mapping visible and survives any row
that got in with different casing.

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0791aac42465"
down_revision: Union[str, Sequence[str], None] = "63f6c85ccfce"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.alter_column(
        "catalog_column",
        "is_nullable",
        existing_type=sa.TEXT(),
        type_=sa.Boolean(),
        existing_nullable=True,
        schema="gsf",
        postgresql_using=(
            "CASE"
            " WHEN is_nullable IS NULL THEN NULL"
            " WHEN upper(btrim(is_nullable)) IN ('NO', 'FALSE', 'F', 'N', '0')"
            " THEN false"
            " WHEN upper(btrim(is_nullable)) IN ('YES', 'TRUE', 'T', 'Y', '1')"
            " THEN true"
            " ELSE NULL"
            " END"
        ),
    )


def downgrade() -> None:
    """Downgrade schema.

    Back to ``information_schema``'s vocabulary rather than ``'true'``/
    ``'false'``, so a downgraded catalog reads the way the old code expects.
    NULL stays NULL — it means "undetermined", which is not the same as "YES".
    """
    op.alter_column(
        "catalog_column",
        "is_nullable",
        existing_type=sa.Boolean(),
        type_=sa.TEXT(),
        existing_nullable=True,
        schema="gsf",
        postgresql_using=(
            "CASE"
            " WHEN is_nullable IS NULL THEN NULL"
            " WHEN is_nullable THEN 'YES'"
            " ELSE 'NO'"
            " END"
        ),
    )
