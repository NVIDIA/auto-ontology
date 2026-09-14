"""Add rules, and the source of a tag

Adds ``rule`` -- a saved global search, as the term, the match option and the
filters it replays -- and ``rule__tag``, the tags it applies to everything that
search matches. Adds to ``tag_target`` the two columns that say *who or what*
applied each label, which is what a rule needs in order to be un-appliable.

``filters`` is ``jsonb`` rather than a column per flag. It is written by the
create and read back to be handed to the search, never filtered or joined on,
and ``GlobalSearchFilters`` gains a field whenever the search does, which would
otherwise be a migration per flag.

``created_by`` is ``NOT NULL``, unlike ``tag.created_by``: the route refuses a
create it cannot attribute, because a rule goes on labelling the catalog long
after it was saved. ``modified_by`` is nullable beside it, as a tag's is -- a
rule nobody has renamed has no editor, which is the same fact as ``modified``
still equalling ``created``. Neither is a foreign key, for the reasons the tag
columns are not: the accounts live in ``frontend."user"``, which Alembic does
not manage, and a rule must outlive the account that saved it.

``uq_rule_name_lower`` makes a rule name unique, folded for case and for
surrounding space, exactly as ``uq_tag_name_lower`` does for a tag: the name is
how the settings list refers to a rule, and one name naming two of them would
leave a reader unable to say which they are about to delete.

``rule__tag`` keys on ``(rule_id, tag_id)`` and cascades from both sides, so a
tag deleted from the vocabulary leaves the rules that applied it rather than
leaving them pointing at an id that names nothing. ``position`` records the
order the tags were picked in, which the primary key does not, and is all the
table holds beyond the two ids: everything else about a tag is read through the
foreign key, so nothing here can disagree with the vocabulary.

On ``tag_target``, ``tagged_by`` is the account that clicked, as a Better Auth
user id from the gateway's identity header, and ``rule_id`` is the rule that
matched when a rule applied the label rather than a person. ``rule_id``
cascades, so deleting a rule takes back what it labelled: a rule-applied tag is
the rule still holding rather than a fact of its own, and would otherwise
outlive the only thing that could explain it. Both are nullable, and every row
that existed before this migration has both null -- there was nothing recording
a source to backfill from, and a label whose source is unknown is what the page
shows for one.

Revision ID: e37f4a1f2f5d
Revises: 00a48e8b426f
Create Date: 2026-09-08 10:46:23.656124

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "e37f4a1f2f5d"
down_revision: Union[str, Sequence[str], None] = "00a48e8b426f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "rule",
        sa.Column(
            "id",
            sa.Text(),
            server_default=sa.text("(gen_random_uuid())::text"),
            nullable=False,
        ),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("search_term", sa.Text(), nullable=False),
        sa.Column(
            "text_match_option",
            sa.Text(),
            server_default=sa.text("'contains'"),
            nullable=False,
        ),
        sa.Column(
            "filters",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
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
        sa.Column("created_by", sa.Text(), nullable=False),
        sa.Column("modified_by", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_rule")),
    )
    op.create_index(
        "uq_rule_name_lower",
        "rule",
        [sa.literal_column("lower(TRIM(BOTH FROM name))")],
        unique=True,
    )
    op.create_index(
        "ix_rule_name_lower", "rule", [sa.literal_column("lower(name)")], unique=False
    )
    op.create_table(
        "rule__tag",
        sa.Column("rule_id", sa.Text(), nullable=False),
        sa.Column("tag_id", sa.Text(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["rule_id"],
            ["rule.id"],
            name=op.f("fk_rule__tag_rule_id_rule"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["tag_id"],
            ["tag.id"],
            name=op.f("fk_rule__tag_tag_id_tag"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("rule_id", "tag_id", name=op.f("pk_rule__tag")),
    )
    op.create_index("ix_rule__tag_tag_id", "rule__tag", ["tag_id"], unique=False)

    op.add_column("tag_target", sa.Column("tagged_by", sa.Text(), nullable=True))
    op.add_column("tag_target", sa.Column("rule_id", sa.Text(), nullable=True))
    op.create_index("ix_tag_target_rule_id", "tag_target", ["rule_id"], unique=False)
    op.create_foreign_key(
        op.f("fk_tag_target_rule_id_rule"),
        "tag_target",
        "rule",
        ["rule_id"],
        ["id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    """Downgrade schema.

    ``tag_target`` first: its foreign key is what would otherwise stop ``rule``
    from being dropped, and the labels themselves stay -- losing the source of a
    label is this migration going away, while losing the label is not.
    """
    op.drop_constraint(
        op.f("fk_tag_target_rule_id_rule"), "tag_target", type_="foreignkey"
    )
    op.drop_index("ix_tag_target_rule_id", table_name="tag_target")
    op.drop_column("tag_target", "rule_id")
    op.drop_column("tag_target", "tagged_by")

    op.drop_index("ix_rule__tag_tag_id", table_name="rule__tag")
    op.drop_table("rule__tag")
    op.drop_index("ix_rule_name_lower", table_name="rule")
    op.drop_index("uq_rule_name_lower", table_name="rule")
    op.drop_table("rule")
