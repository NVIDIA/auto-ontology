# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Who may edit a row. The stored owner does not decide it."""

from auto_ontology.dal.owners import CATALOG_EDIT
from auto_ontology.server.owners import decide_edit, normalize_owner_id

#: A permission a viewer does not hold, so the tests can tell a granted
#: permission from a role that simply lacks it.
_WITHOUT = "zone:manage"


def test_admin_can_edit() -> None:
    assert decide_edit(role="admin", permission=CATALOG_EDIT)
    assert decide_edit(role="admin", permission=_WITHOUT)


def test_a_viewer_cannot_edit_the_catalog() -> None:
    assert not decide_edit(role="viewer", permission=CATALOG_EDIT)
    assert not decide_edit(role="viewer", permission=_WITHOUT)


def test_a_missing_role_cannot_edit() -> None:
    assert not decide_edit(role=None, permission=CATALOG_EDIT)


def test_blank_owner_id_means_unassigned() -> None:
    assert normalize_owner_id("  ") is None
    assert normalize_owner_id(" user-1 ") == "user-1"
    assert normalize_owner_id(None) is None
