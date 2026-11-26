# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Abstract base class for conversation sources.

The transcript counterpart of
:class:`auto_ontology.connectors.base.SQLDatabase`, and deliberately the same
shape: a fixed set of questions, answered per format, returning one agreed
structure so that nothing downstream branches on where the text came from.

**Why this exists at all, when SQL needs no equivalent.** A database answers
questions about itself -- ``get_tables`` is a query, not a parse -- so a
connector only has to translate dialect. A transcript cannot be asked
anything. It is a file whose layout is known to the person who exported it and
to nobody else, so the knowledge a connector gets for free from
``information_schema`` has to be written out by hand here. The Slack
line-start regex is this package's ``SELECT ... FROM sqlite_master``.

The consequence is that the format must be *named*. A connection string
carries its scheme and the registry reads it; a JSON file of turns carries
nothing, so the caller states which source it holds.

Example
-------
::

    from auto_ontology.conversations.sources import create_source

    source = create_source("slack", root="conversation/data/slack/v1")
    for conversation_id in source.get_conversations():
        conversation = source.read(conversation_id)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from auto_ontology.conversations.model import Conversation, Turn

__all__ = ["ConversationSource"]


class ConversationSource(ABC):
    """One transcript format, read into turns.

    Implementations share no code -- pairing alternating roles and grouping
    Slack threads have nothing in common, any more than ``sqlite_master`` and
    ``pg_class`` do. What they share is the output: a list of :class:`Turn`
    numbered across the conversation, which ingest consumes without knowing
    which implementation produced it.
    """

    def __init__(self, root: str | Path) -> None:
        #: Where the exported transcripts live. The counterpart of a
        #: connection string: the one piece of deployment-specific
        #: information a source needs, and the only constructor argument,
        #: so the registry can build any source uniformly.
        self.root = Path(root)

    @property
    @abstractmethod
    def source(self) -> str:
        """The registry key, e.g. ``"chat"`` or ``"slack"``.

        Reported rather than inferred so a caller holding an instance can say
        what it is -- the counterpart of
        :attr:`~auto_ontology.connectors.base.SQLDatabase.dialect`.
        """

    @abstractmethod
    def get_conversations(self) -> list[str]:
        """Every conversation id available, in a stable order.

        Names only: no transcript is read. The counterpart of ``get_tables``,
        and used the same way -- to enumerate before committing to the cost of
        reading any one of them.
        """

    @abstractmethod
    def get_turns(self, conversation_id: str) -> list[Turn]:
        """The whole parse for one conversation.

        ``index`` must run from 0 across the conversation without gaps, since
        it is what every citation and every "the turn after this one" resolves
        against. ``session_index`` marks the sitting -- whatever the format's
        natural boundary is, a sitting for a chat log and a calendar day for a
        channel -- because that boundary is what makes a changed value a change
        of mind rather than a correction.
        """

    def read(self, conversation_id: str) -> Conversation:
        """``conversation_id`` as a :class:`Conversation`, ready to ingest."""
        return Conversation(
            conversation_id=conversation_id,
            turns=tuple(self.get_turns(conversation_id)),
        )
