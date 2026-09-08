"""add tags

Adds ``tag`` and the ``tag_target`` link table that attaches a Tag to exactly
one of the five taggable entities, and tightens the two PROPERTY_OF link
tables to one Term per attribute.

``tag_target`` carries one nullable FK per taggable entity with a
``num_nonnulls(...) = 1`` check, so a row names a single target, and a
per-entity ``UNIQUE(tag_id, <entity>_id)`` so the same Tag lands on a target
once.

``tag.created_by`` and ``tag.modified_by`` hold the Better Auth user ids the
Next.js gateway forwards. Plain nullable ``text`` and not foreign keys: the
accounts live in ``frontend."user"``, a schema Prisma owns and Alembic does not
manage, so there is nothing here to reference -- and a tag has to outlive the
account that made it, which a FK with any ``ondelete`` would either forbid or
quietly rewrite. Nullable because the gateway's header is not guaranteed:
FastAPI is reachable directly on the private network, and a caller that skips
the gateway still creates tags, with no author to record.

``UNIQUE(attribute_id)`` on the PROPERTY_OF link tables makes a database rule
out of a convention the writers keep. Both tables key on
``(attribute_id, term_id)``, which only forbids the same pair twice -- one
attribute could be a property of two Terms. Nothing wanted that:
``link_to_term`` deletes any other link before inserting, a ColumnAttribute's
``term_name`` is part of its merge key, and every read joins through the link,
so a second Term would show one attribute twice on pages that mean to list it
once. Applies as-is on a store that has been written to only through the DAL.
A duplicate would fail here rather than be dropped, because picking which of
two Terms to keep is not a decision a migration can make.

Revision ID: 00a48e8b426f
Revises: 8c3d5b17a204
Create Date: 2026-09-03 15:57:08.113113

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "00a48e8b426f"
down_revision: Union[str, Sequence[str], None] = "8c3d5b17a204"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "tag",
        sa.Column(
            "id",
            sa.Text(),
            server_default=sa.text("(gen_random_uuid())::text"),
            nullable=False,
        ),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column(
            "created",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "modified",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("created_by", sa.Text(), nullable=True),
        sa.Column("modified_by", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_tag")),
    )
    op.create_index(
        "uq_tag_name_lower",
        "tag",
        [sa.literal_column("lower(trim(name))")],
        unique=True,
    )
    op.create_table(
        "tag_target",
        sa.Column(
            "id",
            sa.Text(),
            server_default=sa.text("(gen_random_uuid())::text"),
            nullable=False,
        ),
        sa.Column("tag_id", sa.Text(), nullable=False),
        sa.Column("term_id", sa.Text(), nullable=True),
        sa.Column("table_id", sa.Text(), nullable=True),
        sa.Column("column_id", sa.Text(), nullable=True),
        sa.Column("column_attribute_id", sa.Text(), nullable=True),
        sa.Column("sql_attribute_id", sa.Text(), nullable=True),
        sa.Column(
            "tagged",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "num_nonnulls(term_id, table_id, column_id, column_attribute_id, sql_attribute_id) = 1",
            name=op.f("ck_tag_target_exactly_one_target"),
        ),
        sa.ForeignKeyConstraint(
            ["column_attribute_id"],
            ["column_attribute.id"],
            name=op.f("fk_tag_target_column_attribute_id_column_attribute"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["column_id"],
            ["catalog_column.id"],
            name=op.f("fk_tag_target_column_id_catalog_column"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["sql_attribute_id"],
            ["sql_attribute.id"],
            name=op.f("fk_tag_target_sql_attribute_id_sql_attribute"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["table_id"],
            ["catalog_table.id"],
            name=op.f("fk_tag_target_table_id_catalog_table"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["tag_id"],
            ["tag.id"],
            name=op.f("fk_tag_target_tag_id_tag"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["term_id"],
            ["term.id"],
            name=op.f("fk_tag_target_term_id_term"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_tag_target")),
        sa.UniqueConstraint(
            "tag_id",
            "column_attribute_id",
            name="uq_tag_target_tag_id_column_attribute_id",
        ),
        sa.UniqueConstraint(
            "tag_id", "column_id", name="uq_tag_target_tag_id_column_id"
        ),
        sa.UniqueConstraint(
            "tag_id", "sql_attribute_id", name="uq_tag_target_tag_id_sql_attribute_id"
        ),
        sa.UniqueConstraint("tag_id", "table_id", name="uq_tag_target_tag_id_table_id"),
        sa.UniqueConstraint("tag_id", "term_id", name="uq_tag_target_tag_id_term_id"),
    )
    op.create_index(
        "ix_tag_target_column_attribute_id",
        "tag_target",
        ["column_attribute_id"],
        unique=False,
    )
    op.create_index(
        "ix_tag_target_column_id", "tag_target", ["column_id"], unique=False
    )
    op.create_index(
        "ix_tag_target_sql_attribute_id",
        "tag_target",
        ["sql_attribute_id"],
        unique=False,
    )
    op.create_index("ix_tag_target_table_id", "tag_target", ["table_id"], unique=False)
    op.create_index("ix_tag_target_term_id", "tag_target", ["term_id"], unique=False)
    op.create_unique_constraint(
        op.f("uq_column_attribute__term_attribute_id"),
        "column_attribute__term",
        ["attribute_id"],
    )
    op.create_unique_constraint(
        op.f("uq_sql_attribute__term_attribute_id"),
        "sql_attribute__term",
        ["attribute_id"],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint(
        op.f("uq_sql_attribute__term_attribute_id"),
        "sql_attribute__term",
        type_="unique",
    )
    op.drop_constraint(
        op.f("uq_column_attribute__term_attribute_id"),
        "column_attribute__term",
        type_="unique",
    )
    op.drop_index("ix_tag_target_term_id", table_name="tag_target")
    op.drop_index("ix_tag_target_table_id", table_name="tag_target")
    op.drop_index("ix_tag_target_sql_attribute_id", table_name="tag_target")
    op.drop_index("ix_tag_target_column_id", table_name="tag_target")
    op.drop_index("ix_tag_target_column_attribute_id", table_name="tag_target")
    op.drop_table("tag_target")
    op.drop_index("uq_tag_name_lower", table_name="tag")
    op.drop_table("tag")
