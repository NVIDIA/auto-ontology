# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Path resolution for SQLite connection strings."""

from pathlib import Path

import pytest

from auto_ontology.connectors.sqlite import (
    SQLiteDatabase,
    _sqlite_path_from_connection_string,
)


@pytest.mark.parametrize(
    ("connection_string", "expected_path"),
    [
        pytest.param(
            "sqlite:///C:/data/db.sqlite",
            "C:/data/db.sqlite",
            id="uri-drive-letter",
        ),
        pytest.param(
            "sqlite:C:/data/db.sqlite",
            "C:/data/db.sqlite",
            id="uri-slashless-drive-letter",
        ),
        pytest.param(
            "C:/data/db.sqlite",
            "C:/data/db.sqlite",
            id="bare-drive-path",
        ),
    ],
)
def test_connection_string_resolves_to_db_file(
    connection_string: str, expected_path: str
) -> None:
    assert _sqlite_path_from_connection_string(connection_string) == Path(expected_path)


def test_uri_with_windows_drive_letter_connects(tmp_path: Path) -> None:
    db_file = tmp_path / "t.sqlite"
    db_file.write_bytes(b"")
    # ``f"sqlite:///{path}"`` is how callers build URIs from a pathlib path; on
    # Windows the path embeds the drive letter and must still resolve to the file.
    db = SQLiteDatabase(f"sqlite:///{db_file}")
    try:
        assert db.database_name == "t"
    finally:
        db.close()
