# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transcript formats, read into turns.

One module per format, selected by name through :func:`create_source`, in the
same arrangement as :mod:`auto_ontology.connectors`.
"""

from auto_ontology.conversations.sources.base import ConversationSource
from auto_ontology.conversations.sources.chat import ChatSource
from auto_ontology.conversations.sources.registry import SOURCE_REGISTRY, create_source
from auto_ontology.conversations.sources.slack import SlackSource

__all__ = [
    "SOURCE_REGISTRY",
    "ChatSource",
    "ConversationSource",
    "SlackSource",
    "create_source",
]
