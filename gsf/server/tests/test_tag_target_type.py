# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The API's tag-target vocabulary must match the DAL's.

``TagTargetType`` spells the five kinds of taggable object as literals instead
of importing the DAL's ``TARGET_*`` constants, so ``gsf/server/models.py`` --
which is nothing but the shape of the API -- does not drag SQLAlchemy and the
whole table metadata in behind five strings.

The cost of that is two spellings of one vocabulary, and the drift would
otherwise be quiet: FastAPI would accept a ``type`` the DAL has no column for
and the request would reach ``_target_column`` and fail there, answering 500
where the enum exists to answer 422 and list what it does accept. These tests
are what makes it loud, and they are the reason the literals are allowed.

Needs no database: both sides are module-level constants.
"""

from __future__ import annotations

from gsf.dal.tags import (
    TARGET_COLUMN,
    TARGET_COLUMN_ATTRIBUTE,
    TARGET_SQL_ATTRIBUTE,
    TARGET_TABLE,
    TARGET_TERM,
)
from gsf.server.models import TagTargetType


def test_the_two_vocabularies_are_the_same_five_strings() -> None:
    assert {member.value for member in TagTargetType} == {
        TARGET_TERM,
        TARGET_TABLE,
        TARGET_COLUMN,
        TARGET_COLUMN_ATTRIBUTE,
        TARGET_SQL_ATTRIBUTE,
    }


def test_each_member_carries_its_own_dal_constant() -> None:
    """Member by member, not just as a set.

    Swapping two values -- ``TERM = "table"`` and ``TABLE = "term"`` -- leaves
    the set identical and sends every tagged term to the wrong column.
    """
    assert TagTargetType.TERM == TARGET_TERM
    assert TagTargetType.TABLE == TARGET_TABLE
    assert TagTargetType.COLUMN == TARGET_COLUMN
    assert TagTargetType.COLUMN_ATTRIBUTE == TARGET_COLUMN_ATTRIBUTE
    assert TagTargetType.SQL_ATTRIBUTE == TARGET_SQL_ATTRIBUTE
