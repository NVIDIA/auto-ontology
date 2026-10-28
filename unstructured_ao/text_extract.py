# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plain-text extraction for the automatic unstructured ingest route."""

from pathlib import Path
from typing import Any

#: Characters per row: about one PDF page row, well under the embedding
#: model's 4096-token ceiling and the semantic pass's 10k-character window.
MAX_CHUNK_CHARS = 4_000

#: Where a long text is cut, coarsest first.
_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", " ")


def split_text(
    text: str,
    max_chars: int = MAX_CHUNK_CHARS,
    *,
    separators: tuple[str, ...] = _SEPARATORS,
) -> list[str]:
    """*text* in pieces of at most *max_chars*, cut at the coarsest break that fits.

    Paragraphs are packed together while they fit; a paragraph that does not
    is cut at lines, then words, and a single word longer than the limit is
    sliced.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    if not separators:
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]

    separator, finer = separators[0], separators[1:]
    pieces: list[str] = []
    current = ""
    for part in text.split(separator):
        part = part.strip()
        if not part:
            continue
        if len(part) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(split_text(part, max_chars, separators=finer))
            continue
        candidate = f"{current}{separator}{part}" if current else part
        if len(candidate) > max_chars:
            pieces.append(current)
            current = part
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def extract_chunks(file_path: str, client: Any | None = None) -> list[dict[str, Any]]:
    """Read a UTF-8 text file into retriever-shaped records, one per piece."""
    del client
    path = Path(file_path).expanduser().resolve()
    text = path.read_text(encoding="utf-8")
    return [
        {
            "text": piece,
            "source_id": str(path),
            "element_type": "text",
            "page_number": None,
        }
        for piece in split_text(text)
    ]
