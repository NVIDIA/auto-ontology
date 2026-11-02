# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Message builders for gateway-compatible text-to-SQL LLM requests."""

from collections.abc import Iterable

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage


def _join_parts(parts: Iterable[str]) -> str:
    return "\n\n".join(part.strip() for part in parts if part and part.strip())


def single_turn_messages(
    *,
    system_parts: Iterable[str],
    user_parts: Iterable[str],
) -> list[BaseMessage]:
    """Build exactly one system message followed by exactly one user message."""
    return [
        SystemMessage(content=_join_parts(system_parts)),
        HumanMessage(content=_join_parts(user_parts)),
    ]


def message_contents(
    messages: Iterable[BaseMessage],
    message_type: type[BaseMessage],
) -> list[str]:
    """Collect string content for one message role while preserving order."""
    return [
        str(message.content)
        for message in messages
        if isinstance(message, message_type) and str(message.content).strip()
    ]
