# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trigram index on tag.name for global search

``tag`` became a searchable kind of its own -- the global search has a Tags tab
-- which put ``ILIKE '%term%'`` on ``tag.name`` on the same path as every other
catalog match. The baseline builds this index for the nine tables searched at
the time and creates ``pg_trgm`` along with them, so this revision adds the
tenth and nothing else.

One index rather than two: a tag is a name and its authorship, with no
description column to match against. ``auto_ontology.dal.schema`` skips the
missing column when it builds these, and
``auto_ontology.dal.search._LABELS_WITHOUT_DESCRIPTION`` is the same fact on the
query side.

Revision ID: 47ffb7499dfa
Revises: c778a7d94c20
Create Date: 2026-09-30 11:29:27.072077

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "47ffb7499dfa"
down_revision: Union[str, Sequence[str], None] = "c778a7d94c20"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_index(
        "ix_tag_name_trgm",
        "tag",
        ["name"],
        unique=False,
        postgresql_using="gin",
        postgresql_ops={"name": "gin_trgm_ops"},
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_tag_name_trgm", table_name="tag")
