# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from fastapi import HTTPException

from gsf.server.chat import router


@pytest.fixture
def catalog_databases(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        router.datasources_dal,
        "fetch_databases",
        lambda zone_ids=None: [
            {
                "id": "42d62c22-b1c4-4ed7-8173-6aec9e2bd432",
                "name": "wideworldimporters",
            }
        ],
    )


def test_resolves_catalog_database_uuid_to_name(catalog_databases: None) -> None:
    assert (
        router._resolve_chat_target_db("42d62c22-b1c4-4ed7-8173-6aec9e2bd432")
        == "wideworldimporters"
    )


def test_resolves_database_name_case_insensitively(
    catalog_databases: None,
) -> None:
    assert router._resolve_chat_target_db("WIDEWORLDIMPORTERS") == "wideworldimporters"


def test_rejects_unknown_target_database(catalog_databases: None) -> None:
    with pytest.raises(HTTPException) as exc_info:
        router._resolve_chat_target_db("unknown")

    assert exc_info.value.status_code == 422
    assert "does not match a catalog database UUID or name" in exc_info.value.detail
