# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Message construction helpers for focused SQL-generation calls."""

from collections.abc import Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage


def build_focused_sql_messages(
    existing_messages: Sequence[BaseMessage],
    *,
    system_prompt: str,
    user_prompt: str,
) -> list[BaseMessage]:
    """Build one system message and one request message for SQL generation.

    Earlier user and assistant turns are deliberately excluded: the rendered
    request already contains the resolved question and any relevant conversation
    examples. Existing system context is retained after the SQL policy so global
    caller instructions still apply without breaking the stable SQL-policy prefix.
    """

    inherited_system_context = [
        message.content.strip()
        for message in existing_messages
        if isinstance(message, SystemMessage)
        and isinstance(message.content, str)
        and message.content.strip()
    ]
    system_sections = [system_prompt.strip(), *inherited_system_context]
    return [
        SystemMessage(content="\n\n".join(system_sections)),
        HumanMessage(content=user_prompt),
    ]
