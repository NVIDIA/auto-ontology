# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ReAct agent's ``grep`` tool: search a ``run_sql`` result too long to show.

``run_sql`` shows the first rows of a result. When there are more, it keeps the
result as a page, and ``grep`` lets the model check the rest — whether a value
is there, what a later row holds — without a second query or a context full of
rows it does not need.
"""

from __future__ import annotations

import time

import regex

from auto_ontology.retrieval.text_to_sql.react import config
from auto_ontology.retrieval.text_to_sql.react.context import ToolContext
from auto_ontology.retrieval.text_to_sql.react.run_sql import _cell, _clip


def grep(ctx: ToolContext, page: int, pattern: str = "", offset: int = 0) -> str:
    """Rows of saved *page* matching *pattern*, from match *offset* on.

    *pattern* is a case-insensitive regular expression, or plain text when it
    does not compile; an empty one matches every row, which pages through the
    result. Each row is shown with its 1-based number in the result.

    The pattern comes from the model, so matching runs under a time budget
    (:func:`config.grep_timeout_s`): a pattern that backtracks without end is
    reported as too slow instead of stalling the episode.
    """
    if not ctx.pages:
        return (
            "Error: no saved results. A run_sql result is saved as a page only "
            "when it is too long to show."
        )
    try:
        page = int(page)
        offset = max(int(offset or 0), 0)
    except (TypeError, ValueError):
        return "Error: page and offset must be numbers."
    if not 1 <= page <= len(ctx.pages):
        return f"Error: no page {page}; pages 1 to {len(ctx.pages)} exist."

    frame = ctx.pages[page - 1]
    rows = list(frame.itertuples(index=False))
    lines = [" | ".join(_cell(v) for v in row) for row in rows]
    # Shown cut short, but matched whole: a value past the cut is still there.
    full = [" | ".join("NULL" if v is None else str(v) for v in row) for row in rows]
    pattern = (pattern or "").strip()
    if pattern:
        try:
            compiled = regex.compile(pattern, regex.IGNORECASE)
        except regex.error:
            compiled = regex.compile(regex.escape(pattern), regex.IGNORECASE)
        deadline = time.monotonic() + config.grep_timeout_s()
        hits: list[tuple[int, str]] = []
        try:
            for n, (text, line) in enumerate(zip(full, lines), 1):
                # The budget is shared by every row: once it is spent the search
                # stops, rather than giving each later row a fresh sliver.
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError
                if compiled.search(text, timeout=left):
                    hits.append((n, line))
        except TimeoutError:
            return (
                f"Error: the pattern {pattern!r} took too long to match. Use a "
                "simpler pattern or plain text."
            )
        found = f"{len(hits)} of {len(lines)} rows match"
    else:
        hits = list(enumerate(lines, 1))
        found = f"{len(lines)} rows"
    if not hits:
        return f"0 of {len(lines)} rows match {pattern!r}."

    shown = hits[offset : offset + config.max_result_rows()]
    if not shown:
        return f"{found}; offset {offset} is past the last one."
    summary = found
    if len(shown) < len(hits):
        summary += f", {offset + 1} to {offset + len(shown)} shown"
    header = "row | " + " | ".join(str(c) for c in frame.columns)
    body = [f"{n} | {line}" for n, line in shown]
    return _clip("\n".join([summary, header, *body]))
