# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A Slack channel export: threads, many speakers, no roles.

Everything that makes this format hard is in the ``text`` field. A thread
arrives as one blob with its messages run together, so the first job is
finding where each one starts -- and only about half the lines begin a
message, the rest being continuations of the one above. Get that wrong and
one person's standup is split across five speakers.

Three things follow from the channel being a workplace rather than a product
chat, and each is handled here because each is a property of the export and
not of any one dataset built from it:

* **Posts with no author.** A Slack Workflow app posts on someone's behalf and
  leaves the account field empty, writing the name into the body instead.
* **Secrets.** People paste branch passwords, CI tokens and meeting links into
  channels. Redacted at the parse, which is the last point before the text
  reaches a store, an index and eventually an answer.
* **Threads that start before their root.** The export leads with a message
  that is not always the earliest one in the thread.

The pairing into turns is delegated to
:func:`~auto_ontology.conversations.model.turns_from_sessions` rather than
reimplemented, so a Slack turn is assembled by the same code as a chat turn
and the two cannot drift.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from auto_ontology.conversations.model import Turn, turns_from_sessions
from auto_ontology.conversations.sources.base import ConversationSource

__all__ = ["SlackSource", "redact"]

#: Where the channel directories sit under the root. The raw export is kept
#: beside what is derived from it -- the hash that pins it, the question set
#: written against it -- so the root is the revision and this is the export
#: within it.
_SUBDIR = "export"

#: Every line of ``text`` that starts a message. The rest are continuations of
#: the message above -- 87,556 of the 192,387 non-blank lines in one export,
#: mostly the bullet lists in standup updates -- and attaching them to their
#: own message is the whole job of the parse. Measured: every one of the 9,316
#: threads opens on a line this matches.
_MESSAGE = re.compile(
    r"^(?P<reply>Reply )?"
    r"(?P<stamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) "
    r"(?P<author>.+?):[ ]?(?P<body>.*)$"
)

#: The author of a Slack Workflow post, written into the body because the app
#: posted it and the account field is empty. Trailing rules are part of the
#: form's own layout, not the name.
_UPDATE_AUTHOR = re.compile(r"^:update:\s*(?P<name>.+?)\s*(?:-{3,}\s*)?$")

#: What the workflow app posts to open a standup. Nobody wrote it, so it has no
#: author to recover and nothing to ask about.
_REMINDER = "please post your daily update"

#: What an unresolved author arrives as.
_UNKNOWN = "unknown user"

#: Something said in the channel that is a secret rather than a fact about
#: one. A graph built over an export would otherwise embed these, return them
#: from retrieval and quote them in answers.
_SECRET = re.compile(
    r"(?P<label>passwords?|passwd|pwd|passcode|api[_ ]?key|secret|token)"
    # A few words may sit between the label and the separator: "password for
    # reminder is ...". Over-redacting a word that was never a secret costs
    # nothing; missing one that was costs a rotation.
    r"(?P<gap>(?:\s+\w+){0,3}\s*(?:is|are|=|:)\s*)"
    r"[`\"']?(?P<value>[^\s`\"'<>,;()]{6,80})",
    re.I,
)

#: The same thing hidden in a link: ``?token=...``, ``?pwd=...``.
_SECRET_PARAM = re.compile(
    r"(?P<label>[?&](?:token|pwd|password|access_token|api_key)=)"
    r"(?P<value>[^\s&|>]{6,80})",
    re.I,
)

#: Words that follow "password" without being one.
_NOT_A_SECRET = frozenset(
    {
        "required",
        "rotated",
        "changed",
        "updated",
        "expired",
        "invalid",
        "missing",
        "correct",
        "protected",
        "needed",
        "reset",
        "the",
        "not",
    }
)

_REDACTED = "[REDACTED]"


def redact(text: str) -> str:
    """*text* with disclosed credentials replaced.

    Applied as each message is rendered, the one place every turn's text
    passes through, so nothing downstream holds a secret even though the
    export still does. Emoji shortcodes are the trap worth naming: ``still
    need the token :pensive:`` matches the shape of a disclosure exactly,
    which is why a value ending in a colon is left alone.
    """

    def swap(match: re.Match[str]) -> str:
        value = match.group("value")
        if value.endswith(":") or value.lower() in _NOT_A_SECRET:
            return match.group(0)
        return f"{match.group('label')}{match.group('gap')}{_REDACTED}"

    text = _SECRET.sub(swap, text)
    return _SECRET_PARAM.sub(lambda m: f"{m.group('label')}{_REDACTED}", text)


@dataclass(frozen=True)
class Message:
    """One message inside a thread, with its continuation lines attached."""

    is_reply: bool
    stamp: str
    author: str
    body: str

    def render(self) -> str:
        """Back to the export's own wording, with the author as resolved.

        Rebuilt rather than sliced out of ``text`` because the author may have
        been recovered since, and a turn that still reads ``Unknown User``
        cannot be asked who said it.
        """
        prefix = "Reply " if self.is_reply else ""
        body = redact(self.body)
        return f"{prefix}{self.stamp} {self.author}: {body}".rstrip()


def parse_messages(text: str) -> list[Message]:
    """Split a thread's ``text`` into its messages, in order.

    A line that is not a message header belongs to the message above it: the
    standup form emits a heading and a bullet list per update, and splitting
    those into messages of their own would attribute one person's update to
    several speakers and leave most of them without a name at all.
    """
    messages: list[Message] = []
    pending: list[str] = []

    def flush() -> None:
        if messages and pending:
            body = "\n".join([messages[-1].body, *pending]).rstrip()
            messages[-1] = Message(
                messages[-1].is_reply,
                messages[-1].stamp,
                messages[-1].author,
                body,
            )
        pending.clear()

    for line in text.split("\n"):
        match = _MESSAGE.match(line)
        if match is None:
            # Blank lines are kept only between content: they separate the
            # sections of a standup update, and collapsing them runs the
            # headings into the bullets.
            if line.strip() or pending:
                pending.append(line)
            continue
        flush()
        messages.append(
            Message(
                is_reply=bool(match.group("reply")),
                stamp=match.group("stamp"),
                author=match.group("author").strip(),
                body=match.group("body"),
            )
        )
    flush()
    return messages


def resolve_author(messages: list[Message]) -> list[Message] | None:
    """Name the workflow app's post from its body, or drop it.

    Returns ``None`` for the app's own daily reminder, which has no author
    because no person wrote it and nothing to extract because it says the same
    sentence every day. 2,207 of one export's 2,210 authorless threads are
    workflow posts in the two standup channels; about a quarter of those are
    reminders.
    """
    if not messages:
        return None
    root = messages[0]
    if root.author.strip().lower() != _UNKNOWN:
        return messages

    if _REMINDER in root.body.lower():
        return None

    first_line = root.body.split("\n", 1)[0].strip()
    found = _UPDATE_AUTHOR.match(first_line)
    if found is None:
        # Authorless and not a form we recognise. Kept, because the text is
        # still something that was said in the channel, but no name is
        # invented for it.
        return messages

    # The form's first line is the name itself, so dropping it loses nothing
    # and leaving it in would have the turn open by repeating its own author.
    remainder = root.body.split("\n", 1)[1] if "\n" in root.body else ""
    return [
        Message(
            root.is_reply,
            root.stamp,
            found.group("name").strip(),
            remainder.lstrip("\n"),
        ),
        *messages[1:],
    ]


def thread_day(messages: list[Message]) -> str:
    """The day a thread began, which is not always its root's day.

    In 575 of one export's 5,145 threads with replies the root is stamped
    later than a reply -- the root is the message the export chose to lead
    with, not the first one sent. Dating the session from the root files
    those under the wrong day, which is the one thing a session is for.
    """
    return min(message.stamp for message in messages).split(" ", 1)[0]


def _anchor(day: str) -> str:
    """``2025-09-02`` as ``September-02-2025``, which is BEAM's wording.

    Matched so a prompt or a reader tuned on one corpus is not reading a
    second date format in the other.
    """
    return datetime.strptime(day, "%Y-%m-%d").strftime("%B-%d-%Y")


def _thread_as_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """One thread as the two role-tagged messages that make a single turn.

    **Always two, even when nobody replied.**
    :func:`~auto_ontology.conversations.model.turns_from_messages` joins
    consecutive messages from the same side into one turn, so a thread with no
    replies followed by another thread would come out as a single turn
    carrying both, attributing one person's question to another's thread. An
    empty answer closes the turn instead.

    This function is the compromise the format is forced into: a thread has
    many speakers and no roles, and it is being described to a model that
    takes exactly two. It is the piece that goes away once a turn can hold a
    list of attributed messages.
    """
    root, replies = messages[0], messages[1:]
    return [
        {
            "role": "user",
            "content": root.render(),
            # Every turn carries its own date. BEAM states one on three turns
            # in ninety-four and the library carries it forward to the rest;
            # Slack stamps every message, so nothing has to be assumed.
            "time_anchor": _anchor(thread_day(messages)),
        },
        {
            "role": "assistant",
            "content": "\n".join(reply.render() for reply in replies),
        },
    ]


class SlackSource(ConversationSource):
    """``<root>/export/<channel>/<channel>.json``, one file per channel.

    A conversation is a channel and a session is a calendar day. The day is
    the only boundary the format offers, and it carries the meaning a session
    is needed for: a value restated a week later is a revision, while the same
    words half an hour later are a correction.

    The subdirectory is this class's business rather than the caller's, for
    the same reason the dialect is a connector's: a caller that had to name it
    would be a caller that knows one format's layout, and then every ingest
    path grows a branch per source instead of passing a name and a root.
    """

    @property
    def source(self) -> str:
        return "slack"

    def get_conversations(self) -> list[str]:
        directory = self.root / _SUBDIR
        if not directory.is_dir():
            return []
        return sorted(
            path.stem
            for path in directory.glob("*/*.json")
            if path.stem == path.parent.name
        )

    def get_turns(self, conversation_id: str) -> list[Turn]:
        sessions, _ = self._sessions(conversation_id)
        return turns_from_sessions(sessions)

    def thread_metadata(self, conversation_id: str) -> list[dict[str, Any]]:
        """One row per turn: who posted, the link, and the export's summaries.

        Outside the :class:`ConversationSource` contract because it is not
        something every format has, and outside the turn text because the
        titles and summaries are another pipeline's model output -- a graph
        built from them would be recalling claims nobody made. Useful for
        choosing which threads to ask about, never for answering.
        """
        _, metadata = self._sessions(conversation_id)
        return metadata

    def _path(self, conversation_id: str) -> Path:
        return self.root / _SUBDIR / conversation_id / f"{conversation_id}.json"

    def _sessions(
        self, conversation_id: str
    ) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
        """A channel's threads as sessions of one day each, oldest first."""
        payload = json.loads(self._path(conversation_id).read_text(encoding="utf-8"))
        records = payload.get("records") or []
        ordered = sorted(
            records, key=lambda r: float((r.get("source") or {}).get("ts") or 0.0)
        )

        sessions: list[list[dict[str, Any]]] = []
        metadata: list[dict[str, Any]] = []
        current_day: str | None = None

        for record in ordered:
            parsed = resolve_author(parse_messages(record.get("text") or ""))
            if not parsed:
                continue

            day = thread_day(parsed)
            if day != current_day:
                sessions.append([])
                current_day = day
            sessions[-1].extend(_thread_as_messages(parsed))

            source = record.get("source") or {}
            generated = record.get("generated") or {}
            metadata.append(
                {
                    "turn_index": len(metadata),
                    "day": day,
                    "author": parsed[0].author,
                    "reply_count": len(parsed) - 1,
                    "thread_link": source.get("thread_link") or "",
                    "thread_title": generated.get("thread_title") or "",
                    "thread_summary": generated.get("thread_summary") or "",
                    "issue_resolved": generated.get("issue_resolved") or "",
                }
            )
        return sessions, metadata
