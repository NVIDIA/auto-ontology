# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plain-text extraction for the automatic unstructured ingest route."""

from pathlib import Path
from typing import Any


def extract_chunks(file_path: str, client: Any | None = None) -> list[dict[str, Any]]:
    """Read a UTF-8 text file into one retriever-shaped record."""
    del client
    path = Path(file_path).expanduser().resolve()
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    return [
        {
            "text": text,
            "source_id": str(path),
            "element_type": "text",
            "page_number": None,
        }
    ]
