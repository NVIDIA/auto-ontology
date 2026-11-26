# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select a conversation source by name.

The counterpart of
:func:`auto_ontology.connectors.registry.create_connector`, with one
difference worth stating because it is the whole reason this package exists:
a connector's identity is discovered, read off the connection string's scheme,
while a transcript's has to be declared. There is nothing in a JSON file of
messages that says whether its speakers are roles or people.
"""

from __future__ import annotations

from pathlib import Path

from auto_ontology.conversations.sources.base import ConversationSource
from auto_ontology.conversations.sources.chat import ChatSource
from auto_ontology.conversations.sources.slack import SlackSource

__all__ = ["SOURCE_REGISTRY", "create_source"]

#: Every transcript format that can be read, by the name callers pass.
SOURCE_REGISTRY: dict[str, type[ConversationSource]] = {
    "chat": ChatSource,
    "slack": SlackSource,
}


def create_source(source: str, root: str | Path) -> ConversationSource:
    """Build the source named *source*, reading transcripts under *root*."""
    source_class = SOURCE_REGISTRY.get(source.strip().lower())
    if source_class is None:
        known = ", ".join(sorted(SOURCE_REGISTRY))
        raise ValueError(f"Unknown conversation source {source!r}. Known: {known}.")
    return source_class(root)
