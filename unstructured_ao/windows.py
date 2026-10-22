# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Group a document's chunks into LLM-sized extraction windows.

Extraction walks these windows in order so the whole document is read, not
just the prefix that happens to fit one prompt.  The window is also the unit
of provenance: it carries the chunk ids of the text the model was shown, which
is what lets an extracted attribute be linked back to the chunk that states it
without a vector search.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import unstructured_ao.config as config

#: Quotes shorter than this match too many chunks to identify one.
_MIN_QUOTE_CHARS = 24


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


@dataclass(frozen=True)
class Window:
    """A contiguous run of chunks small enough to extract from in one call."""

    chunk_ids: list[str]
    texts: list[str]

    @property
    def text(self) -> str:
        """The window's chunks concatenated, without labels."""
        return "\n\n".join(self.texts)

    @property
    def labelled_text(self) -> str:
        """The window as the LLM sees it: one ``[n]``-prefixed block per chunk.

        Labels are 1-based positions rather than chunk uuids because a model
        copies ``[4]`` reliably and mangles ``a3f8c1d2-…``; :meth:`resolve`
        maps them back, so a hallucinated label never reaches the database.
        """
        return "\n\n".join(f"[{i}] {t}" for i, t in enumerate(self.texts, 1))

    def resolve(self, label: int) -> str | None:
        """Chunk id for a 1-based *label*, or ``None`` if out of range."""
        if 1 <= label <= len(self.chunk_ids):
            return self.chunk_ids[label - 1]
        return None

    def find_quote(self, quote: str) -> str | None:
        """Chunk id whose text literally contains *quote*, if exactly one does.

        A verbatim hit proves the link independently of the label the model
        returned, so it is the one piece of attribution that cannot be wrong.
        """
        needle = _normalise(quote or "")
        if len(needle) < _MIN_QUOTE_CHARS:
            return None
        matches = [
            cid
            for cid, text in zip(self.chunk_ids, self.texts)
            if needle in _normalise(text)
        ]
        return matches[0] if len(matches) == 1 else None


def build_windows(
    chunk_ids: list[str],
    texts: list[str],
    *,
    max_chars: int = config.EXTRACT_WINDOW_CHARS,
) -> list[Window]:
    """Pack *texts* into windows of at most *max_chars*, in document order.

    Chunks are never split across windows: a chunk is the smallest thing an
    attribute can be linked to, so half of one is not a usable citation.  A
    single chunk over the limit therefore becomes a window of its own.
    """
    windows: list[Window] = []
    cur_ids: list[str] = []
    cur_texts: list[str] = []
    cur_chars = 0

    for chunk_id, raw in zip(chunk_ids, texts):
        text = (raw or "").strip()
        if not text:
            continue
        if cur_ids and cur_chars + len(text) > max_chars:
            windows.append(Window(cur_ids, cur_texts))
            cur_ids, cur_texts, cur_chars = [], [], 0
        cur_ids.append(chunk_id)
        cur_texts.append(text)
        cur_chars += len(text)

    if cur_ids:
        windows.append(Window(cur_ids, cur_texts))
    return windows
