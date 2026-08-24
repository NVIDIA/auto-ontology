# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Postgres persistence and bounded history loading for chat conversations."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

import psycopg
from psycopg.rows import dict_row

from gsf.infra.postgres import FRONTEND_SCHEMA, get_postgres_connection_string

logger = logging.getLogger(__name__)

_MAX_HISTORY_TURNS = 5
_MAX_HISTORY_CHARS = 12_000
_MAX_QUESTION_CHARS = 2_000
_MAX_RESPONSE_CHARS = 4_000
_MAX_SQL_CHARS = 6_000


class ConversationAccessError(Exception):
    """Raised when a conversation exists but belongs to another user."""


@dataclass(frozen=True)
class ConversationTurn:
    """One completed user/assistant exchange suitable for LLM context."""

    question: str
    response: str
    sql_code: str | None = None


@dataclass(frozen=True)
class PreparedConversation:
    """Identifiers and prior history created when accepting a chat turn."""

    history: list[ConversationTurn]
    analytics_id: str


def _is_context_assistant(row: dict) -> bool:
    if row["role"] != "assistant" or row["sql_response"] is not None:
        return False
    content = (row["content"] or "").strip()
    if content.startswith("```chart") or content.startswith("```chart-carousel"):
        return False
    if content.startswith("Agent failed:") or content.startswith(
        "Agent stream failed:"
    ):
        return False
    return bool(content or row["sql_code"])


def _rows_to_history(rows: list[dict]) -> list[ConversationTurn]:
    turns: list[ConversationTurn] = []
    pending_question: str | None = None

    for row in rows:
        role = row["role"]
        if role == "user":
            pending_question = (row["content"] or "").strip()
            continue
        if pending_question and _is_context_assistant(row):
            turns.append(
                ConversationTurn(
                    question=pending_question[:_MAX_QUESTION_CHARS],
                    response=(row["content"] or "").strip()[:_MAX_RESPONSE_CHARS],
                    sql_code=(row["sql_code"] or "")[:_MAX_SQL_CHARS] or None,
                )
            )
            pending_question = None

    selected: list[ConversationTurn] = []
    used_chars = 0
    for turn in reversed(turns):
        turn_chars = len(turn.question) + len(turn.response) + len(turn.sql_code or "")
        if selected and used_chars + turn_chars > _MAX_HISTORY_CHARS:
            break
        selected.append(turn)
        used_chars += turn_chars
        if len(selected) >= _MAX_HISTORY_TURNS:
            break
    return list(reversed(selected))


def prepare_conversation(
    *,
    conversation_id: uuid.UUID,
    user_id: str,
    question: str,
    source: str,
) -> PreparedConversation:
    """Verify/create a conversation, load prior turns, then persist the user turn."""

    conversation_str = str(conversation_id)
    analytics_id = str(uuid.uuid4())
    with psycopg.connect(get_postgres_connection_string(), connect_timeout=3) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                f"""
                INSERT INTO {FRONTEND_SCHEMA}.conversations (id, user_id, title, created_at, updated_at)
                VALUES (%s, %s, %s, NOW(), NOW())
                ON CONFLICT (id) DO NOTHING
                RETURNING user_id
                """,
                (conversation_str, user_id, question[:50] or "New conversation"),
            )
            owner = cur.fetchone()
            if owner is None:
                cur.execute(
                    f"SELECT user_id FROM {FRONTEND_SCHEMA}.conversations WHERE id = %s FOR UPDATE",
                    (conversation_str,),
                )
                owner = cur.fetchone()
            if owner is None or owner["user_id"] != user_id:
                raise ConversationAccessError(conversation_str)

            cur.execute(
                f"""
                SELECT role, content, sql_code, sql_response
                FROM {FRONTEND_SCHEMA}.messages
                WHERE conversation_id = %s
                ORDER BY created_at ASC, id ASC
                """,
                (conversation_str,),
            )
            history = _rows_to_history([dict(row) for row in cur.fetchall()])

            cur.execute(
                f"""
                INSERT INTO {FRONTEND_SCHEMA}.messages
                    (id, conversation_id, role, content, created_at)
                VALUES (%s, %s, 'user', %s, NOW())
                """,
                (str(uuid.uuid4()), conversation_str, question),
            )
            cur.execute(
                f"""
                INSERT INTO {FRONTEND_SCHEMA}.conversation_analytics
                    (id, user_id, source, question, question_timestamp)
                VALUES (%s, %s, %s, %s, NOW())
                """,
                (analytics_id, user_id, source, question),
            )
            cur.execute(
                f"UPDATE {FRONTEND_SCHEMA}.conversations SET updated_at = NOW() WHERE id = %s",
                (conversation_str,),
            )

    return PreparedConversation(history=history, analytics_id=analytics_id)


def persist_assistant_result(
    *,
    conversation_id: uuid.UUID,
    user_id: str,
    analytics_id: str,
    response: str,
    sql_code: str | None,
) -> None:
    """Persist one final assistant turn and complete its analytics row."""

    conversation_str = str(conversation_id)
    with psycopg.connect(get_postgres_connection_string(), connect_timeout=3) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT 1 FROM {FRONTEND_SCHEMA}.conversations WHERE id = %s AND user_id = %s",
                (conversation_str, user_id),
            )
            if cur.fetchone() is None:
                raise ConversationAccessError(conversation_str)
            cur.execute(
                f"""
                INSERT INTO {FRONTEND_SCHEMA}.messages
                    (id, conversation_id, role, content, sql_code, created_at)
                VALUES (%s, %s, 'assistant', %s, %s, NOW())
                """,
                (str(uuid.uuid4()), conversation_str, response, sql_code),
            )
            cur.execute(
                f"""
                UPDATE {FRONTEND_SCHEMA}.conversation_analytics
                SET response = %s, sql = %s, response_timestamp = NOW()
                WHERE id = %s AND user_id = %s
                """,
                (response, sql_code, analytics_id, user_id),
            )
            cur.execute(
                f"UPDATE {FRONTEND_SCHEMA}.conversations SET updated_at = NOW() WHERE id = %s",
                (conversation_str,),
            )


def persist_result_message(
    *,
    conversation_id: uuid.UUID,
    user_id: str,
    content: str,
    sql_response: str | None,
) -> None:
    """Persist the chart/table bubble ("Message 2") for a completed turn.

    Called unconditionally from ``_pump`` so this bubble is guaranteed to
    exist — same guarantee ``persist_assistant_result`` already gives the
    SQL answer — instead of depending on a browser tab staying around to
    ask for it.
    """

    conversation_str = str(conversation_id)
    with psycopg.connect(get_postgres_connection_string(), connect_timeout=3) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT 1 FROM {FRONTEND_SCHEMA}.conversations "
                "WHERE id = %s AND user_id = %s",
                (conversation_str, user_id),
            )
            if cur.fetchone() is None:
                raise ConversationAccessError(conversation_str)
            cur.execute(
                f"""
                INSERT INTO {FRONTEND_SCHEMA}.messages
                    (id, conversation_id, role, content, sql_response, created_at)
                VALUES (%s, %s, 'assistant', %s, %s, NOW())
                """,
                (str(uuid.uuid4()), conversation_str, content, sql_response),
            )
            cur.execute(
                f"UPDATE {FRONTEND_SCHEMA}.conversations "
                "SET updated_at = NOW() WHERE id = %s",
                (conversation_str,),
            )


def create_stateless_analytics(*, user_id: str, question: str, source: str) -> str:
    """Create analytics for an authenticated one-shot request."""

    analytics_id = str(uuid.uuid4())
    with psycopg.connect(get_postgres_connection_string(), connect_timeout=3) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {FRONTEND_SCHEMA}.conversation_analytics
                    (id, user_id, source, question, question_timestamp)
                VALUES (%s, %s, %s, %s, NOW())
                """,
                (analytics_id, user_id, source, question),
            )
    return analytics_id


def persist_analytics_result(
    *,
    analytics_id: str,
    user_id: str,
    response: str,
    sql_code: str | None,
) -> None:
    """Complete an analytics row for a stateless authenticated request."""

    with psycopg.connect(get_postgres_connection_string(), connect_timeout=3) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE {FRONTEND_SCHEMA}.conversation_analytics
                SET response = %s, sql = %s, response_timestamp = NOW()
                WHERE id = %s AND user_id = %s
                """,
                (response, sql_code, analytics_id, user_id),
            )


__all__ = [
    "ConversationAccessError",
    "ConversationTurn",
    "PreparedConversation",
    "create_stateless_analytics",
    "persist_analytics_result",
    "persist_assistant_result",
    "persist_result_message",
    "prepare_conversation",
]
