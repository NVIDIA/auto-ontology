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

import re

from auto_ontology.retrieval.text_to_sql.react import config
from auto_ontology.retrieval.text_to_sql.react.context import ToolContext
from auto_ontology.retrieval.text_to_sql.react.run_sql import _cell, _clip


def grep(ctx: ToolContext, page: int, pattern: str = "", offset: int = 0) -> str:
    """Rows of saved *page* matching *pattern*, from match *offset* on.

    *pattern* is a case-insensitive regular expression, or plain text when it
    does not compile; an empty one matches every row, which pages through the
    result. Each row is shown with its 1-based number in the result.
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
    lines = [" | ".join(_cell(v) for v in row) for row in frame.itertuples(index=False)]
    pattern = (pattern or "").strip()
    if pattern:
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error:
            regex = re.compile(re.escape(pattern), re.IGNORECASE)
        hits = [(n, line) for n, line in enumerate(lines, 1) if regex.search(line)]
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
