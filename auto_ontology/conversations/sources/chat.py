# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two-party chat logs: a user asks, an assistant answers.

The format BEAM ships and the one every assistant transcript arrives in. Its
defining property is that the speaker is a *role* rather than a person, and
that there are exactly two of them, alternating -- which is what lets a turn
be a question and an answer rather than a list of messages.

The pairing itself lives in :mod:`auto_ontology.conversations.model` and is
not repeated here. This module is the thin part: find the file, read the
sessions out of it, and hand them over. Keeping the pairing where it is
matters more than tidiness -- it is the code every BEAM measurement was taken
through, and re-implementing it behind a new interface would put a silent
rewrite underneath a number we compare against.
"""

from __future__ import annotations

import json
from typing import Any

from auto_ontology.conversations.model import Turn, turns_from_sessions
from auto_ontology.conversations.sources.base import ConversationSource

__all__ = ["ChatSource"]

#: Transcripts live one JSON file per conversation, named for it. The same
#: layout BEAM's download writes and the one the eval reads back.
_SUBDIR = "conversations"


class ChatSource(ConversationSource):
    """``<root>/conversations/<id>.json``, each holding role-tagged sessions."""

    @property
    def source(self) -> str:
        return "chat"

    def get_conversations(self) -> list[str]:
        directory = self.root / _SUBDIR
        if not directory.is_dir():
            return []
        return sorted(path.stem for path in directory.glob("*.json"))

    def get_turns(self, conversation_id: str) -> list[Turn]:
        payload = self._payload(conversation_id)
        return turns_from_sessions(payload["chat"])

    def _payload(self, conversation_id: str) -> dict[str, Any]:
        path = self.root / _SUBDIR / f"{conversation_id}.json"
        return json.loads(path.read_text(encoding="utf-8"))
