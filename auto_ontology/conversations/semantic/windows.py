# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Group a transcript's turns into LLM-sized extraction windows.

Extraction walks these in order so the whole conversation is read, not the
prefix that happens to fit one prompt. A BEAM transcript runs to 94 turns and
tens of thousands of characters; the quota that gets revised, the bug that gets
fixed and the deadline that moves are spread across all of it, and a prefix
holds the first of each and none of the revisions.

The window is also the unit of provenance. It carries the turn indexes of the
text the model was shown, which is what lets an extracted attribute be linked
back to the turn that states it without a vector search.

**Blocks are labelled with their real turn index**, unlike the document
counterpart, which numbers blocks from one per window. A turn index is already
a small integer a model copies reliably, it is what the transcript's own
ordering means, and it is what
:func:`~auto_ontology.conversations.store.fetch_turns` takes as a span -- so a
citation needs no translation to become a read.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

__all__ = ["Window", "build_windows"]

#: Quotes shorter than this match too many turns to identify one.
_MIN_QUOTE_CHARS = 24

#: How much turn text one extraction call is shown. Large enough that a BEAM
#: session is usually one window -- a revision and the value it revises tend to
#: fall in the same sitting, and splitting them costs the `supersedes` edge that
#: is the reason for reading at all.
WINDOW_CHARS = 12000


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


@dataclass(frozen=True)
class Window:
    """A contiguous run of turns small enough to extract from in one call."""

    #: Turn indexes, in transcript order. These are the labels the model sees.
    turn_indexes: list[int]
    #: Rendered text per turn, parallel to *turn_indexes*.
    texts: list[str]
    #: ``time_anchor`` per turn, parallel to the above. Carried so the extractor
    #: can date an event without the caller joining back to the store.
    anchors: list[str | None]

    @property
    def text(self) -> str:
        """The window's turns concatenated, without labels."""
        return "\n\n".join(self.texts)

    @property
    def labelled_text(self) -> str:
        """The window as the LLM sees it: one ``[turn n]``-prefixed block each.

        The date is on the label rather than left in the turn text. Transcripts
        state the day once when a sitting opens and never again, so most turns
        carry no date of their own; an extractor asked to order events from the
        text alone has nothing to order them by on 91 turns out of 94.
        """
        blocks = []
        for index, body, anchor in zip(self.turn_indexes, self.texts, self.anchors):
            dated = f"[turn {index}, {anchor}]" if anchor else f"[turn {index}]"
            blocks.append(f"{dated} {body}")
        return "\n\n".join(blocks)

    def resolve(self, label: int) -> int | None:
        """*label* if this window holds that turn, else ``None``.

        A model that cites a turn outside the window it was shown has either
        remembered an earlier window or invented the number, and neither is
        evidence this window supports.
        """
        return label if label in set(self.turn_indexes) else None

    def find_quote(self, quote: str) -> int | None:
        """Turn index whose text literally contains *quote*, if exactly one does.

        A verbatim hit proves the link independently of the label the model
        returned, so it is the one piece of attribution that cannot be wrong.
        """
        needle = _normalise(quote or "")
        if len(needle) < _MIN_QUOTE_CHARS:
            return None
        matches = [
            index
            for index, body in zip(self.turn_indexes, self.texts)
            if needle in _normalise(body)
        ]
        return matches[0] if len(matches) == 1 else None


def turn_block(question: str, answer: str) -> str:
    """One turn as the extractor reads it.

    Both sides, labelled. Which party said a thing decides what it is: a
    preference and a constraint are only ever the user's, and crediting the
    assistant's suggestion to the user is how a transcript acquires an opinion
    nobody held.
    """
    parts = []
    if question:
        parts.append(f"user: {question}")
    if answer:
        parts.append(f"assistant: {answer}")
    return "\n".join(parts)


def build_windows(
    turns: Sequence[dict[str, Any]],
    *,
    max_chars: int = WINDOW_CHARS,
) -> list[Window]:
    """Pack *turns* into windows of at most *max_chars*, in transcript order.

    *turns* are rows as
    :func:`~auto_ontology.conversations.store.fetch_turns` returns them.

    Turns are never split across windows: a turn is the smallest thing an
    attribute can be linked to, so half of one is not a usable citation. A
    single turn over the limit therefore becomes a window of its own.

    Session boundaries are not respected, deliberately. The edges worth having
    cross them -- a value revised in session 2 supersedes the one stated in
    session 1 -- and a window that stops at the boundary shows the model one
    side of every such pair.
    """
    windows: list[Window] = []
    indexes: list[int] = []
    texts: list[str] = []
    anchors: list[str | None] = []
    chars = 0

    for turn in turns:
        body = turn_block(
            str(turn.get("question") or "").strip(),
            str(turn.get("answer") or "").strip(),
        )
        if not body:
            continue
        if indexes and chars + len(body) > max_chars:
            windows.append(Window(indexes, texts, anchors))
            indexes, texts, anchors, chars = [], [], [], 0
        indexes.append(int(turn["turn_index"]))
        texts.append(body)
        anchor = turn.get("time_anchor")
        anchors.append(str(anchor) if anchor else None)
        chars += len(body)

    if indexes:
        windows.append(Window(indexes, texts, anchors))
    return windows
