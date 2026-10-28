# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collapse the property keys one transcript used for a single measurement.

:mod:`derive` recognises a revised value by an exact ``attribute_key`` match, so
two readings of one quantity filed under two keys are two unrelated statements
and the revision is simply missing. It happens: the extractor sees each window
in turn and names the property from the words that window used, and "sort step
latency" in turn 19 becomes "API response time" by turn 75.

Measured on one BEAM transcript, that cost two real revisions out of eight --
an application latency going 180ms to 140ms, and a daily API ceiling going
3,000 to 4,800 calls a day.

Why a model and not a similarity threshold. Among the keys one subject carries,
the pairs that must merge are the lexically distant ones and the pairs that must
stay apart are the near-identical ones:

    daily_quota / rate_limit_per_day            merge, share nothing lexically
    rate_limit_per_day / rate_limit_per_minute  keep apart, one word different
    api_response_time / page_load_time          keep apart, same shape

A distance threshold gets all three backwards. One call per transcript against
the hundred extraction already pays is not the part of the bill worth saving.

Candidates are found without the model: only keys a *single subject* carries are
ever offered, which is what keeps the question small and keeps two subjects'
measurements from being merged on the strength of sharing a word.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

__all__ = ["reconcile_keys"]


_SYSTEM = """\
You decide which property keys from one conversation name the SAME measurable
property of the same thing. Keys were invented one turn at a time, from the
words that turn happened to use, so one figure that was measured twice often
ended up under two names -- and until they are merged, the graph cannot say the
figure ever changed.

The test: could a person write one true sentence of the form "the <subject>'s
<property> went from <first value> to <second value>"? Then merge. Expect to
find these -- a conversation that measures something twice usually renames it.

MERGE, because one figure is being read twice under two names:
  sort_time_ms 180ms  +  carrier_reply_time 140ms      (one app, one latency,
      named for the sort step early on and for the carrier call later)
  daily_label_cap 4800/day  +  rate_limit_per_day 3000/day (one API, one daily
      ceiling, called a cap once and a rate limit once)
  manifest_coverage 52%  +  coverage_current 71%       (one suite, one figure)

KEEP APART:
  carrier_reply_time  +  page_render_time     two different measurements that
      are both times
  rate_limit_per_day  +  rate_limit_per_minute  one ceiling per window, two
      windows, both true at once
  latency_current  +  latency_target          a reading and a goal; the gap
      between them is not the figure moving
  archive_size  +  label_image_size           a part is not the whole
  queue_wait  +  carrier_reply_time           on one feature: what the user
      feels and what the server took are measured separately and differ by the
      work in between. Likewise render time, round-trip time, time to first
      byte -- one feature has many latencies and they are not one figure.

The names being similar is not evidence either way: the first two merges above
share no words, and the second rejection differs by one. Read what is being
measured, not how it is spelled.

Each reading is shown with the sentence it was taken from. Those sentences
decide the hard cases, and they outrank the key names whenever the two
disagree. Two readings whose sentences describe different parts of one thing
are two properties, however alike their keys read; two readings whose sentences
describe the same part are one property, however differently their keys are
spelled. When a sentence is missing, judge on the key and the value alone.
"""

_TEMPLATE = """\
Conversation properties, grouped by the thing they are about. Each line is
`key | value | stated in turn(s)`.

{subjects}

Return the groups of two or more keys that name one property. Use only keys
listed above, each in at most one group, and never group keys from two
different subjects.
"""


class KeyGroup(BaseModel):
    """Keys the model says are one property."""

    keys: list[str] = Field(
        default_factory=list,
        description="Two or more keys, exactly as listed, naming one property.",
    )
    property_name: str = Field(
        default="",
        description="What the shared property is, in a few words.",
    )


class KeyGroups(BaseModel):
    groups: list[KeyGroup] = Field(default_factory=list)


#: Values that state a limit rather than a reading. "below 180ms" is where the
#: figure was asked to get to, and merging its key into the key of the figure
#: itself makes the distance between a goal and a measurement look like the
#: measurement moved. Checked on the value because the kind cannot be relied on:
#: a transcript's own "reduce the sort step latency, below 180ms" came back
#: tagged ``preference``, which no goal filter catches.
_BOUNDS = (
    "below ",
    "under ",
    "over ",
    "at most ",
    "at least ",
    "no more than ",
    "max ",
    "min ",
    "<",
    ">",
    "~",
)


def _is_bound(value: str) -> bool:
    return value.strip().casefold().startswith(_BOUNDS)


def _valued(statements: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Readings: a key, a value, and a value that is a figure rather than a limit.

    Goals and proposals are dropped on the same terms :func:`derive._revisions`
    drops them, so a key only ever offered here is a key that could chain if it
    were merged. Offering one that could not is pure downside -- the merge
    cannot help, and a wrong merge still costs a real distinction.
    """
    from auto_ontology.conversations.semantic.extract import UNREALISED_KINDS

    unrealised = set(UNREALISED_KINDS)
    return [
        row
        for row in statements
        if (row.get("attribute_key") or "").strip()
        and (value := (row.get("value") or "").strip())
        and not _is_bound(value)
        and not unrealised.intersection(row.get("kinds") or ())
    ]


def _by_subject(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Rows indexed by each subject they are about.

    A statement with several subjects -- a latency is measured on a feature and
    in a browser -- appears under each, because which one makes two keys
    comparable is not knowable here. ``_accept`` requires a group to sit inside
    one subject, so appearing under two only means two chances to be offered.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for subject in row.get("about") or ():
            grouped[subject].append(row)
    return grouped


def _candidates(
    rows: Sequence[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Subjects carrying more than one distinct key, which are the only ones askable."""
    return {
        subject: members
        for subject, members in _by_subject(rows).items()
        if len({(row.get("attribute_key") or "").strip() for row in members}) > 1
    }


def _render(candidates: dict[str, list[dict[str, Any]]]) -> str:
    """Each subject's readings, with the sentence each was taken from.

    The quote is here because the key names alone cannot settle the case this
    pass most needs to get right. Two readings hang off one thing, one from its
    network call and one from its rendering, and ``response_time_ms`` beside
    ``latency_ms`` reads like one property under two names -- which is what this
    is looking for -- until you see that the two sentences are about different
    parts. Merging them would chain two unrelated figures into a revision.

    It is free: ``evidence`` is on the node already, on the draft and on the
    stored row alike, so no turn is fetched to print it.
    """
    blocks = []
    for subject in sorted(candidates):
        lines = [f"## {subject}"]
        seen: set[tuple[str, str]] = set()
        for row in sorted(
            candidates[subject], key=lambda r: r.get("attribute_key") or ""
        ):
            key = (row.get("attribute_key") or "").strip()
            value = (row.get("value") or "").strip()
            if (key, value) in seen:
                continue
            seen.add((key, value))
            turns = ", ".join(str(i) for i in sorted(row.get("turn_indexes") or []))
            lines.append(f"- {key} | {value} | turn {turns or 'unknown'}")
            if quote := (row.get("evidence") or "").strip()[:200]:
                lines.append(f'    said as: "{quote}"')
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _accept(
    groups: Sequence[KeyGroup], candidates: dict[str, list[dict[str, Any]]]
) -> dict[str, str]:
    """Resolve the answer into a key rewrite, rejecting anything not asked about.

    Four ways to be rejected, each of which would otherwise merge two keys the
    question never put together: a key no subject offered, a key in two groups,
    a group of fewer than two keys, and a group whose keys no single subject
    carries. A rejected group is dropped on its own -- unlike cognee's batch
    review, which discards a whole answer, there is nothing here to retry on a
    smaller question.

    The canonical key is the one stated earliest, so the chain reads from the
    name the transcript used first.
    """
    keys_by_subject = {
        subject: {(row.get("attribute_key") or "").strip() for row in members}
        for subject, members in candidates.items()
    }
    first_turn: dict[str, int] = {}
    for members in candidates.values():
        for row in members:
            key = (row.get("attribute_key") or "").strip()
            turns = row.get("turn_indexes") or []
            if turns:
                first_turn[key] = min(first_turn.get(key, min(turns)), min(turns))

    rewrite: dict[str, str] = {}
    claimed: set[str] = set()
    for group in groups:
        keys = [key.strip() for key in group.keys if key.strip()]
        if len(set(keys)) < 2:
            continue
        if claimed.intersection(keys):
            logger.warning("Key group %s reuses an already-grouped key.", keys)
            continue
        if not any(set(keys) <= carried for carried in keys_by_subject.values()):
            logger.warning("Key group %s is not carried by one subject.", keys)
            continue
        canonical = min(set(keys), key=lambda k: (first_turn.get(k, 10**9), k))
        logger.info(
            "Keys %s name one property (%s) -- filing them under %r.",
            sorted(set(keys)),
            group.property_name or "unnamed",
            canonical,
        )
        claimed.update(keys)
        for key in set(keys):
            if key != canonical:
                rewrite[key] = canonical
    return rewrite


def reconcile_keys(statements: Sequence[dict[str, Any]]) -> int:
    """Rewrite ``attribute_key`` in place so one property has one key. Returns how many.

    Runs after the subjects are attached and before derivation, which is the only
    window where both are true: the subject is what bounds the question, and the
    keys have to be settled before anything is grouped by them.

    Any failure leaves every key as it was. A transcript whose keys drifted is
    the behaviour of every run before this one, so falling back to it costs the
    revisions it would have recovered and nothing that already worked.
    """
    rows = _valued(statements)
    candidates = _candidates(rows)
    if not candidates:
        return 0

    from auto_ontology.utils.llm_invoke import (
        get_llm_client,
        invoke_with_structured_output,
    )

    try:
        answer = invoke_with_structured_output(
            get_llm_client(temperature=0.0, max_tokens=2048),
            [
                SystemMessage(content=_SYSTEM),
                HumanMessage(content=_TEMPLATE.format(subjects=_render(candidates))),
            ],
            KeyGroups,
        )
    except Exception:
        logger.exception(
            "Could not reconcile the property keys of %d statement(s) -- a "
            "measurement named two ways stays two unrelated readings.",
            len(rows),
        )
        return 0
    if answer is None:
        return 0

    rewrite = _accept(answer.groups, candidates)
    if not rewrite:
        return 0

    changed = 0
    for row in statements:
        key = (row.get("attribute_key") or "").strip()
        if key in rewrite:
            row["attribute_key"] = rewrite[key]
            changed += 1
    logger.info(
        "Reconciled %d key(s) into %d canonical name(s).",
        changed,
        len(set(rewrite.values())),
    )
    return changed
