# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Search a chat transcript that has been ingested as conversational memory.

The read side of :mod:`auto_ontology.conversations`. Ingest puts one vector per
question/answer pair into ``vdb.conversation``; this finds the pairs that bear
on a question and hands them back **in the order they were said**, which is
not the order a similarity search returns them in.

    from auto_ontology.retrieval.conversation import search_turns

    hits = search_turns(
        "when does my first sprint end?",
        database_name="beam_100K_1",
        top_k=10,
        widen=1,
    )
"""

from __future__ import annotations

from auto_ontology.retrieval.conversation.search import TurnHit, search_turns

__all__ = ["TurnHit", "search_turns"]
