# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM extraction of a transcript's knowledge layer.

One pass over the transcript, window by window, producing the graph's nodes and
its ``knowledge`` edges. The ``semantic`` edges come afterwards, from
:mod:`derive` and :mod:`relate`, because they are about how the nodes stand to
one another and need the whole set in view.

A **node** is anything the transcript talks about, on two axes:

*type* is the domain word -- ``Person``, ``Project``, ``Technology``,
``Timestamp``. Open vocabulary, because the extractor's own word for a thing is
better than whichever of five listed types fits worst.

*kinds* is what the node is for, and is multi-valued. ``object`` is a thing that
asserts nothing: ``Redis`` is not a fact, the fact is that the project is built
with it, and that is an edge. The other four -- ``fact``, ``preference``,
``event``, ``state`` -- are statements, and a node is honestly more than one at
once: a deadline has a date (``event``) and moved (``state``).

A **knowledge edge** is a typed relation between two nodes, named with the
transcript's own verb. Closed vocabularies are wrong here: the relation name
*is* the content of the edge.

Statements carry a ``key`` and a ``value`` as well as a sentence. That pairing
is what makes conflict detection free rather than adjudicated: two statements
about one subject with one key and different values are two readings of one
quantity by construction, and no model has to be asked whether they are about
the same thing.

Windows, not one call per turn. A chunk-per-turn extraction cannot see a thing
named in turn 5 and measured in turn 6, and every cross-turn relation would be
lost.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from auto_ontology.conversations.semantic.windows import Window

logger = logging.getLogger(__name__)

__all__ = [
    "NODE_KINDS",
    "STATEMENT_KINDS",
    "EdgeDraft",
    "GraphDraft",
    "NodeDraft",
    "extract_graph",
    "node_text",
    "normalise_name",
    "normalise_role",
    "parse_date",
]

#: Node kinds, as the check constraint spells them.
NODE_KINDS = (
    "object",
    "fact",
    "preference",
    "event",
    "state",
    "target",
    "proposed",
    "request",
)

#: The kinds that assert something, as opposed to naming a thing.
STATEMENT_KINDS = (
    "fact",
    "preference",
    "event",
    "state",
    "target",
    "proposed",
    "request",
)

#: Statements about something other than how the world currently is.
#:
#: Both were measured as errors before they were kinds. Asked which settings
#: the project *enforced*, where the transcript has the user saying "here's an
#: example of what I've tried, I'm not sure this is the right approach", the
#: extractor stored the config as a ``fact`` and the reader recited it as
#: enforced -- the answer the benchmark wants is that nothing was. And a stated
#: goal of 60ms stored as a ``state`` alongside a measured 180ms is two
#: readings of one property, which derivation dutifully turned into a revision
#: of a number nobody ever measured.
#:
#: ``request`` joins them for the same reason: asking for error handling to be
#: sorted out is not a reading of how it currently behaves, and comparing the
#: two by key would date the request against the behaviour.
UNREALISED_KINDS = ("target", "proposed", "request")

NodeKind = Literal[
    "object", "fact", "preference", "event", "state", "target", "proposed", "request"
]


# ---------------------------------------------------------------------------
# What the model returns
# ---------------------------------------------------------------------------


class NodeItem(BaseModel):
    """One node of the graph."""

    name: str = Field(
        description=(
            "The node's name, 1-8 words. For a thing, as the transcript names "
            "it: 'Shipping Rates API', 'Alex', 'Redis'. For a statement, a "
            "short readable summary: 'Daily label cap 4,800 parcels'. For a "
            "date, ISO form: '2023-07-14'."
        )
    )
    type: str = Field(
        description=(
            "The domain word for what this is, capitalised: 'Person', "
            "'Project', 'Technology', 'Feature', 'Metric', 'Timestamp', "
            "'Platform', 'Deadline', 'Constraint'. Use 'Timestamp' for a date."
        )
    )
    kinds: list[NodeKind] = Field(
        description=(
            "What this node is for. Use ['object'] for a THING that asserts "
            "nothing on its own (a person, a library, a platform, a bare "
            "date). Use one or more of 'fact', 'preference', 'event', 'state' "
            "for a STATEMENT. A node can be several: a deadline that moved is "
            "['event', 'state']. Never mix 'object' with the others."
        )
    )
    description: str = Field(
        default="",
        description=(
            "For a thing, one short clause saying what it is. For a statement, "
            "one sentence stating what the speaker said, INCLUDING the value, "
            "date or constraint itself. This is read instead of the "
            "transcript, so a description that omits the figure is useless."
        ),
    )
    key: str = Field(
        default="",
        description=(
            "The MEASURABLE PROPERTY this node states a reading of, lowercase "
            "with underscores: 'daily_label_cap', 'sort_time_ms', "
            "'archive_size_limit'. Two turns stating the same property MUST use "
            "the same key, even when the values differ -- that is how a "
            "revision is detected. Set it ONLY together with 'value', and "
            "leave BOTH empty otherwise: a key naming this node rather than a "
            "property it shares with another reading is worse than no key."
        ),
    )
    value: str = Field(
        default="",
        description=(
            "The reading itself, as stated, with its unit: '4800 parcels/day', "
            "'95ms', '3.1MB'. Short, not a sentence. Set it only together "
            "with 'key'. Empty whenever nothing measurable was stated."
        ),
    )
    occurred_on: str = Field(
        default="",
        description=(
            "Events only. The ISO date the event HAPPENED ('2023-07-14'), read "
            "from the sentence itself. Empty if the sentence does not say "
            "when. NEVER the date of the turn."
        ),
    )
    source_turns: list[int] = Field(
        default_factory=list,
        description="Turn numbers in this window that state or mention it.",
    )
    evidence: str = Field(
        default="",
        description=(
            "A short quote (<=200 characters) copied EXACTLY, character for "
            "character, from one of the labelled turns."
        ),
    )


class EdgeItem(BaseModel):
    """One knowledge relation between two nodes."""

    source: str = Field(description="Name of the node the edge starts at.")
    target: str = Field(description="Name of the node the edge points to.")
    role: str = Field(
        description=(
            "The verb, lowercase with underscores: 'builds', 'built_with', "
            "'has_component', 'deployed_to', 'starts_at', 'ends_at', "
            "'covers', 'restricted_to'. Name what the transcript actually "
            "says. Not 'about' or 'measured_by': what a statement is about is "
            "decided in a later step that can see every thing at once."
        )
    )
    source_turns: list[int] = Field(
        default_factory=list,
        description="Turn numbers in this window that state the relation.",
    )
    evidence: str = Field(
        default="",
        description=(
            "A short quote (<=200 chars) copied EXACTLY from one of those turns."
        ),
    )


class GraphExtractionResult(BaseModel):
    """Everything one window yields."""

    nodes: list[NodeItem] = Field(default_factory=list)
    edges: list[EdgeItem] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# What accumulates across windows
# ---------------------------------------------------------------------------


@dataclass
class NodeDraft:
    """A node accumulated across the windows that mention it."""

    name: str
    type: str
    kinds: list[str] = field(default_factory=list)
    description: str = ""
    key: str = ""
    value: str = ""
    occurred_on: date | None = None
    turn_indexes: set[int] = field(default_factory=set)
    #: Subset of ``turn_indexes`` confirmed by a verbatim quote.
    verified_turn_indexes: set[int] = field(default_factory=set)
    evidence: str = ""
    #: The date on the earliest turn cited. When this was *said*.
    time_anchor: str | None = None
    #: Embedding of :func:`node_text`, reused by the vector write.
    vector: list[float] | None = None

    @property
    def is_statement(self) -> bool:
        return any(kind in STATEMENT_KINDS for kind in self.kinds)


@dataclass
class EdgeDraft:
    """A knowledge relation accumulated across the windows that state it."""

    source: str
    target: str
    role: str
    turn_indexes: set[int] = field(default_factory=set)
    #: Subset of ``turn_indexes`` confirmed by a verbatim quote, as a node's is.
    #: An edge is the part of the graph nothing else checks -- every other pass
    #: compares nodes -- so the one attribution that cannot be wrong is worth
    #: keeping on it too.
    verified_turn_indexes: set[int] = field(default_factory=set)
    evidence: str = ""


@dataclass
class GraphDraft:
    """One transcript's extracted graph, before it is written."""

    nodes: list[NodeDraft] = field(default_factory=list)
    edges: list[EdgeDraft] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM = """You build a knowledge graph from a chat transcript.

You are shown ONE WINDOW of a longer conversation as turns labelled
[turn N, date], plus the nodes already found in earlier windows. Extract the
nodes this window adds and the relations between them.

A NODE is anything the conversation talks about. Give each a `type` (the domain
word: Person, Project, Technology, Feature, Metric, Timestamp, Platform) and
`kinds` (what it is for).

  kinds = ['object']  for a THING that asserts nothing by itself: a person, a
                      library, a platform, a feature, a bare date. 'Redis' is
                      an object. The fact that the project uses Redis is an
                      EDGE, not a property of Redis.

  kinds = one or more of these, for a STATEMENT the speaker made:
    fact        asserted once and stays true (the API in use, the language, the
                defaults chosen).
    preference  a STANDING constraint on how the user wants things done, which
                should still apply to a question asked later: "keep it simple",
                "stay under 3.1MB", "no frameworks, plain scripts only". Only
                the user's; never the assistant's suggestion.
    event       something that happened, tied to a date. Put the ISO date the
                event HAPPENED in occurred_on, read from the sentence. If the
                speaker says on August 2 that the token was obtained on July 14,
                occurred_on is 2023-07-14, NOT the turn's date.
    state       the current value of something measurable that can change (a
                cap, a coverage percentage, a sort time).
    target      a value WANTED but not reached: "get sort time down to 60ms",
                "stay under 3.1MB". Never also 'state' -- a goal is not a
                reading, and storing it as one makes the gap between them look
                like the figure changed.
    proposed    something shown, tried or suggested but not adopted: config the
                user pastes with "here's what I've tried, not sure it's right",
                code the assistant offers as an example, a placeholder domain.
                If a later turn says it was adopted, that turn's node is the
                'fact'; this one stays 'proposed'.
    request     a CONCERN the user wants the product to handle: a failure, an
                edge case or a behaviour they want dealt with. "I want to make
                sure I'm handling errors properly", "I want to display friendly
                messages for 404 and 400", "what if the user keeps retrying
                after hitting the rate limit". Name it as the asking, 'Wants
                friendly messages for 404 and 400', not as the thing asked
                about -- a node named for the part stores the sentence as a
                description of the part and the wanting is lost. Keep one
                request per sentence even when it names several parts.
                This kind is NARROW. Asking for help building, implementing,
                reviewing, explaining or improving something is not a request:
                the whole conversation is that, and a kind that holds most of
                the transcript distinguishes nothing. Nor is a question about
                how something works. The test is whether the sentence names
                something that could go wrong or behave badly and asks for it
                to be handled. Not 'preference' (a standing rule that applies
                to a later answer, even when phrased as a want) and not
                'target' (which wants a value, not work). Never the
                assistant's offer, which is 'proposed'.
                A request never REPLACES anything. When the same sentence also
                says what the project uses, wants or measures, extract those
                nodes exactly as you would have without this kind, and the
                request in addition. And deciding a sentence is NOT a request
                never means dropping it: give it the kind it would have had if
                this kind did not exist. Asked for help writing a config, the
                config is still 'proposed' and the rules in it are still the
                content of that node -- the ask being excluded here says
                nothing about the config, and a transcript whose only record
                of a setting is the bare name of the tool it configures cannot
                answer whether the setting was ever adopted.

  A node can be several kinds at once: a deadline that moved is
  ['event', 'state']. Never mix 'object' with the others.

FEATURES. Every CAPABILITY the conversation builds, changes or measures is a
thing in its own right: type 'Feature', kinds ['object'], named for what it does
-- 'Address lookup', 'Label printing', 'Response cache', 'Retry logic',
'Progress indicator', 'Error handling', 'Pagination'. Extract these as readily as you
extract libraries. A conversation about building software talks about its
features constantly, and a graph of only libraries and tools has nowhere to put
anything that was said about them.

A feature is a PART OF THE PRODUCT that persists: something a user could still
be using next month, which can be measured again then and give a different
number. Apply that test literally, because three other things look like
features and are not:
  - NOT the work done to it. 'Address lookup' is a feature; 'Address lookup
    latency optimization' and 'Label render latency optimization' are work, and
    work cannot hold a reading -- a latency belongs to the thing made faster,
    not to the act of making it faster.
  - NOT a reading taken from it. 'Label printing test coverage' and 'Misroute
    rate reduction' are measurements. The feature is the module tested; the
    coverage is a STATEMENT about it. Naming the measurement as a thing puts
    every reading on its own subject, where nothing can be compared to
    anything.
  - Test those two by the name's last word, which is reliable because both
    classes name themselves. A name ending in optimization, reduction,
    improvement, refactor, migration, fix or cleanup is the WORK; a
    name ending in coverage, rate, latency, time, count, size, percentage or
    score is the READING. In both cases drop the last word and whatever
    qualifies it, and the feature is what remains: name that instead, and put
    the rest in a statement about it.
  - NOT the file or symbol holding it. `rateClient.js`, `settings.js` and
    '.env file for the carrier token' are files; if they matter they are
    Technology.
  - Name the capability, not the implementation. The feature is 'Address
    lookup'; `node-cache` and `p-retry` are separate Technology nodes it is
    built_with.

One node per capability, however many aspects of it the window discusses.
'Address lookup', 'Address lookup results UI', 'Address lookup test cases',
'Address lookup API call reduction' and 'Progress indicator for address lookup'
are ONE feature seen from five sides -- list 'Address lookup' and attach the
rest as statements about it. Likewise every cache concern -- expiration,
eviction, locking, compression, what is cached -- is the one 'Response cache'.
When a candidate name only makes sense by naming another feature inside it, it
is that feature.
  - A feature is a thing even when the window only measures it. "The address
    lookup now responds in 95ms" is a statement AND evidence that the Address
    lookup feature exists; list both.
  - Reuse one node per capability across windows, exactly as named before.
    'Address lookup' and 'Address lookup feature' are one feature the graph
    cannot put two measurements on.
  - Expect a long conversation to have on the order of ten or twenty features
    in total, most of them already in the list of nodes found earlier. A window
    that adds five new ones has almost certainly split one capability, or
    promoted a measurement.
  - This matters more than it looks. Every measurement is a property of
    something, and when the feature is missing the whole application absorbs it
    -- the address lookup's 95ms and the app's overall 140ms then read as one
    figure that improved, which is a claim the transcript never made. Split the
    feature too finely and the mirror image happens: the 52% and 71% coverage
    readings land on different subjects and the graph cannot see that one
    replaced the other.

TIMESTAMPS. A point in time is its own node: type 'Timestamp', kinds
['object'], and the normalised date as its name -- '2023', '2023-07',
'2023-07-14'. Keep only the precision the text states; do not invent a day the
speaker did not give.
  - A span with both ends stated ("two weeks from July 14", "between May and
    August") is ONE Timestamp named '<start>/<end>': '2023-07-14/2023-07-28',
    '2023-05/2023-08'. With only one end stated, name that end alone.
  - Reuse one Timestamp node for every reference to the same normalised date.
    Two nodes named '2023-07-14' are one date the graph cannot order.
  - A Timestamp is a leaf. It has no outgoing edges.
  - Link to it with a verb that says what happened then: `obtained_at`,
    `completed_at`, `deployed_at`, `starts_at`, `ends_at`, `runs_during` for a
    span. Not everything is `occurred_at`.
  - Also fill `occurred_on` on the dated statement itself, with the same date.
    Both, not either: the Timestamp node is what a question about a date
    matches, and `occurred_on` is what orders events against each other.

A statement that states a READING of something measurable also carries `key` and
`value`. Both or neither, never one:
  - key is the PROPERTY: daily_label_cap, sort_time_ms, archive_size_limit.
  - value is what it was said to be: "4800 parcels/day", "95ms", "3.1MB".
  - A 'state' or a 'target' always has both -- both are readings of a
    measurable figure, and a 'state' you cannot put a value on is a 'fact'.
  - Everything else usually has NEITHER. "The app handles an address-not-found
    error" is a fact with nothing to measure: leave key and value empty rather
    than inventing 'feature_address_not_found'. A key exists so a SECOND reading
    of the same property can be recognised as the same property; one that will
    only ever name this node does nothing and crowds out the ones that work.
  - Two turns stating the same property MUST use the SAME key even when the
    values disagree. If turn 12 says the cap is 3,000 and turn 48 says it is
    4,800, both are nodes with key 'daily_label_cap' and different values.
    Return BOTH. Do not reconcile them, do not keep only the newest, and do not
    rename the second key to 'updated_daily_label_cap'. Deciding which is
    current is a later step that needs both.
  - The `description` MUST NAME WHAT THE VALUE IS OF. "The user improved sort
    time to 140ms" does not say whose, and a reader holding six latency figures
    will attach it to the wrong one. Write "The dispatch app's overall sort
    time improved to 140ms". This is the ONLY place the
    subject is recorded here, so a description that does not name it loses it:
    write what the figure is a property of even when the turn said only "the
    prototype", and if the turn did not say, write the turn's own words rather
    than guessing.

DENIALS AND RETRACTIONS. A turn saying something did NOT happen, or taking back
something said earlier, is ITS OWN NODE. Never list it as a further citation on
the node it denies.
  - "I've never actually obtained a carrier token" is a new statement, named for
    the denial -- 'Never obtained a carrier token'. It is NOT another turn
    supporting 'Carrier token obtained'. Attaching it there makes the graph say that
    turn CONFIRMS what it refuses, which is the one reading the transcript
    rules out, and it leaves the retraction nowhere at all.
  - This is the one place where reusing an earlier name is wrong. A denial has
    to name what it denies, so it looks exactly like the thing it is about. The
    instruction to reuse names is for a thing MENTIONED again, not for a claim
    WITHDRAWN.
  - If the statement being denied carries a `key`, give the denial the same key
    with the opposite value -- 'yes' against 'no'. Whether something happened
    is a readable property when the transcript disputes it, and a shared key is
    what lets a later step pair the two for free.
  - Do not decide which side is true, and do not drop either. The speaker may
    be mistaken in either direction; returning only the one you find convincing
    is how a conflict stops being visible at all. Deciding is a later step, and
    it needs both.

Do NOT link a statement to what it is about. That edge is decided in a later
step, which sees the whole transcript and the complete list of things; from
inside one window the subject is usually the wrong one, because the only
candidate the window names is whatever happened to be mentioned beside it.

An EDGE joins two THINGS you listed, named with the transcript's own verb:
builds, built_with, has_component, deployed_to, starts_at, ends_at, covers,
restricted_to, and the `_at` verbs above for a Timestamp.

Asking for help with a task is NOT a preference. "Help me fix the timeout
error", "how do I trace this duplicate label", "suggest a load testing tool"
are requests, and the conversation is made almost entirely of them. Record
what the request *established* -- the error exists (fact), the tool they picked
(fact), the constraint they attached to it (preference) -- and not the asking. If you
cannot state a preference as a rule that would change how you answer some
*other* question, it is not one.

Rules:
- Reuse node names EXACTLY as they appear in the list already found. Do not
  rename and do not invent a variant.
- Reuse a `key` EXACTLY as shown for a property already found. A second reading
  of test coverage uses the same key as the first; `coverage_current` and
  `label_test_coverage` for one figure make the two readings uncomparable. Use a
  new key only when the property is genuinely a different measurement, and then
  make the difference explicit in the key itself.
- Every edge's source and target must name a node -- one in this window's list
  or one already found.
- description must contain the actual value, date or constraint. "The user
  discussed the cap" is worthless; "The user's daily label cap is 4,800"
  is the statement.
- source_turns must contain only turn numbers shown in this window, and only
  turns that actually state the thing, not turns that are merely nearby.
- evidence must be copied verbatim from one of those turns. Do not paraphrase,
  summarise, or reformat it; it is checked against the turn text.
- Return empty lists when the window states nothing worth recording.
- Do not invent anything the transcript does not say.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def normalise_name(name: str) -> str:
    """Key under which two spellings of one name collapse."""
    stripped = re.sub(r"[^a-z0-9]+", " ", (name or "").casefold())
    return re.sub(r"\s+", " ", stripped).strip()


def normalise_role(role: str) -> str:
    """An edge role as ``snake_case``."""
    return normalise_name(role).replace(" ", "_")


def node_text(name: str, description: str | None) -> str:
    """The text a node is embedded as.

    One definition, used everywhere, so the vectors a search compares are the
    vectors everything else compared.
    """
    return f"{name}: {description or ''}"


_ISO = re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})")
_MONTHS = {
    month: number
    for number, names in enumerate(
        (
            ("january", "jan"),
            ("february", "feb"),
            ("march", "mar"),
            ("april", "apr"),
            ("may",),
            ("june", "jun"),
            ("july", "jul"),
            ("august", "aug"),
            ("september", "sep", "sept"),
            ("october", "oct"),
            ("november", "nov"),
            ("december", "dec"),
        ),
        start=1,
    )
    for month in names
}
_WORDED = re.compile(
    r"([A-Za-z]{3,9})[\s\-/]+(\d{1,2})(?:st|nd|rd|th)?[\s\-,/]+(\d{4})"
)


def parse_date(value: str | None) -> date | None:
    """A ``date`` from ISO or ``March-10-2024`` form, or ``None``.

    Both forms occur and from different places: the extractor is asked for ISO
    and mostly complies, while BEAM's own per-turn anchors are ``March-15-2024``.
    Parsed here rather than trusted as text because the only thing the column is
    for is arithmetic -- "how many days between" and "which came first" -- and a
    string does neither.
    """
    if not value:
        return None
    text = value.strip()

    found = _ISO.search(text)
    if found:
        year, month, day = (int(part) for part in found.groups())
    else:
        worded = _WORDED.search(text)
        if not worded:
            return None
        name, day_part, year_part = worded.groups()
        month = _MONTHS.get(name.casefold(), 0)
        if not month:
            return None
        day, year = int(day_part), int(year_part)

    try:
        return date(year, month, day)
    except ValueError:
        logger.debug("Unparseable date %r.", value)
        return None


def _clean_kinds(kinds: list[str]) -> list[str]:
    """The node's kinds, deduplicated and with ``object`` kept exclusive.

    A node the model marked both a thing and a statement is one it could not
    decide about. Resolved towards *statement*, because a statement carries a
    value and a thing does not: reading a statement as a thing silently drops
    the figure, and that is the failure that cannot be recovered downstream.
    """
    seen = [kind for kind in dict.fromkeys(kinds) if kind in NODE_KINDS]
    statements = [kind for kind in seen if kind in STATEMENT_KINDS]
    return statements or (["object"] if seen else [])


def _format_known(drafts: Iterable[NodeDraft]) -> str:
    """Nodes earlier windows found, as the vocabulary this window should reuse.

    The ``attribute_key`` is listed, not only the name. Keys drift otherwise:
    one transcript produced ``test_coverage_current``, ``api_test_coverage``
    and ``test_coverage`` for three readings of one figure, and
    :mod:`derive` groups revisions by exact key, so the chain saying which
    reading is current was never built and a reader answered from a date
    instead.
    """
    listed = []
    for draft in drafts:
        key = f" key={draft.key}" if draft.key else ""
        listed.append(f"- {draft.name} [{draft.type}: {'/'.join(draft.kinds)}]{key}")
    return "\n".join(listed) or "(none yet -- this is the first window)"


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def _extract_window(
    window: Window, known: Iterable[NodeDraft]
) -> GraphExtractionResult | None:
    """Extract one window, reusing names from *known*."""
    from auto_ontology.utils.llm_invoke import (
        get_llm_client,
        invoke_with_structured_output,
    )

    return invoke_with_structured_output(
        get_llm_client(temperature=0.0, max_tokens=16384),
        [
            SystemMessage(content=_SYSTEM),
            HumanMessage(
                content=(
                    f"NODES ALREADY FOUND:\n{_format_known(known)}\n\n"
                    f"CONVERSATION WINDOW:\n\n{window.labelled_text}"
                )
            ),
        ],
        GraphExtractionResult,
    )


def _cited_turns(
    window: Window, source_turns: list[int], evidence: str
) -> tuple[set[int], int | None]:
    """Turns of *window* supporting an item, and the quote-verified one.

    A verbatim quote identifies its turn on its own; the numbers the model
    returned are taken only when they fall inside the window. A false citation
    is worse than a missing one, because retrieval cannot tell them apart.
    """
    turns: set[int] = set()
    quoted = window.find_quote(evidence)
    if quoted is not None:
        turns.add(quoted)
    for label in source_turns:
        index = window.resolve(label)
        if index is not None:
            turns.add(index)
    return turns, quoted


def report_unquoted_edges(edges: Sequence[EdgeDraft]) -> list[EdgeDraft]:
    """Log how well the edges are attributed, and return the fabricated ones.

    Three buckets, and only the middle one is a defect:

    ``verified``
        The quote appears verbatim in a turn the edge cites. Nothing else in
        this pipeline establishes that an edge is *true* -- consolidation,
        disambiguation and the subject pass all compare nodes -- so this is the
        only evidence any edge ever carries.
    ``unquoted``
        No quote was returned at all. Not a defect: the extractor omits one
        freely, and absence of evidence is the normal case rather than a claim
        the transcript was misread.
    ``fabricated``
        A quote was returned and **does not appear** in any turn the edge cites.
        That is the one case where the model asserted something checkable and
        the check failed, and it is returned so a caller can act on it.

    Reported rather than dropped here. The buckets have never been measured on
    a real transcript, and an unmeasured filter that deletes edges is how a
    verifier quietly removes a correct graph; the count comes first.
    """
    verified = [edge for edge in edges if edge.verified_turn_indexes]
    quoted = [edge for edge in edges if edge.evidence.strip()]
    fabricated = [edge for edge in quoted if not edge.verified_turn_indexes]

    logger.info(
        "Edge attribution: %d of %d verified by a verbatim quote, %d quoted "
        "without a match, %d with no quote.",
        len(verified),
        len(edges),
        len(fabricated),
        len(edges) - len(quoted),
    )
    for edge in fabricated[:10]:
        logger.warning(
            "Edge %r --%s--> %r cites turn(s) %s with a quote none of them "
            "contains: %r",
            edge.source,
            edge.role,
            edge.target,
            sorted(edge.turn_indexes),
            edge.evidence[:120],
        )
    return fabricated


def _pair_key_and_value(item: NodeItem) -> None:
    """Clear ``key`` and ``value`` unless the item carries both, in place.

    A key earns its place by being reusable: it exists so a *second* reading of
    one property can be recognised as the same property. A key with no value
    names nothing comparable, so it can only ever match itself -- and measured
    on one transcript that was 270 of 349 keyed statements, mostly invented
    labels in the shape of ``feature_city_not_found``. They cost more than the
    empty string would: :func:`_node_key` keys a statement by its property, so
    a per-node key also stops two mentions of that node from collapsing.

    A value with no key is dropped for the same reason from the other side --
    a reading of an unnamed property cannot be compared with anything either.
    Nothing a reader sees is lost in either case: the note renders the name and
    the description, and the prompt already requires the description to carry
    the figure.
    """
    if not (item.key.strip() and item.value.strip()):
        item.key = ""
        item.value = ""


def _node_key(item: NodeItem, kinds: list[str]) -> tuple[str, str, str]:
    """Identity under which two mentions of one node collapse.

    A thing is its name. A statement is its property *and its value*, because
    two readings of one quantity have to stay two nodes: keying a statement by
    property alone is the one mistake that makes a revision unrecoverable --
    the later 4,800 would overwrite the earlier 3,000 and nothing downstream
    could tell the value had ever moved.
    """
    name = normalise_name(item.name)
    if not any(kind in STATEMENT_KINDS for kind in kinds):
        return (name, "", "")
    key = normalise_name(item.key)
    value = normalise_name(item.value)
    return (name if not key else "", key, value or name)


def extract_graph(windows: list[Window]) -> GraphDraft:
    """Read every window and return the transcript's nodes and knowledge edges.

    Windows are read in order and sequentially, not in parallel: each one is
    shown the nodes the earlier ones found so it can reuse a name, and that is
    what keeps one node from becoming four. Parallelising the calls would cost
    more in duplicate nodes than it saves in wall time.
    """
    nodes: dict[tuple[str, str, str], NodeDraft] = {}
    #: Normalised name -> identity, so an edge naming a node by name finds the
    #: row even when that row is keyed by property and value.
    by_name: dict[str, tuple[str, str, str]] = {}
    edges: dict[tuple[str, str, str], EdgeDraft] = {}

    for position, window in enumerate(windows, 1):
        result = _extract_window(window, nodes.values())
        if result is None:
            logger.warning("Window %d returned nothing -- skipped.", position)
            continue

        anchor_by_turn = dict(zip(window.turn_indexes, window.anchors))

        for item in result.nodes:
            kinds = _clean_kinds(list(item.kinds))
            name = normalise_name(item.name)
            if not kinds or not name:
                continue
            # Before the identity, which is derived from the pair.
            _pair_key_and_value(item)
            identity = _node_key(item, kinds)
            draft = nodes.get(identity)
            if draft is None:
                draft = NodeDraft(
                    name=item.name.strip(),
                    type=(item.type or "Thing").strip(),
                    kinds=kinds,
                    description=(item.description or "").strip(),
                    key=normalise_name(item.key).replace(" ", "_"),
                    value=item.value.strip(),
                    occurred_on=parse_date(item.occurred_on),
                )
                nodes[identity] = draft
            else:
                # A later window may see a node as more than the first did --
                # a value it only now calls revisable. Kinds union; the first
                # description wins, because it was written with the turn that
                # introduced the node in view.
                draft.kinds = _clean_kinds(draft.kinds + kinds)
                if not draft.description and item.description:
                    draft.description = item.description.strip()
                if draft.occurred_on is None:
                    draft.occurred_on = parse_date(item.occurred_on)
            by_name.setdefault(name, identity)

            turns, quoted = _cited_turns(window, item.source_turns, item.evidence)
            draft.turn_indexes |= turns
            if quoted is not None:
                draft.verified_turn_indexes.add(quoted)
                if not draft.evidence:
                    draft.evidence = item.evidence.strip()
            # The date of the earliest turn this rests on. Earliest because an
            # event is dated by when it happened and the later turns citing it
            # are usually someone referring back to it.
            dated = [
                anchor_by_turn[i]
                for i in sorted(draft.turn_indexes)
                if anchor_by_turn.get(i)
            ]
            if dated and draft.time_anchor is None:
                draft.time_anchor = dated[0]

        for item in result.edges:
            source, target = normalise_name(item.source), normalise_name(item.target)
            role = normalise_role(item.role)
            # An edge to a node nobody declared is an edge to nothing: the
            # write would have no row to point at, and inventing the node here
            # would create one the transcript never named.
            if not role or source not in by_name or target not in by_name:
                logger.debug(
                    "Edge %r -%s-> %r names an unknown node -- dropped.",
                    item.source,
                    item.role,
                    item.target,
                )
                continue
            if source == target:
                continue
            edge_key = (source, role, target)
            edge = edges.get(edge_key)
            if edge is None:
                edge = EdgeDraft(source=source, target=target, role=role)
                edges[edge_key] = edge
            turns, quoted = _cited_turns(window, item.source_turns, item.evidence)
            edge.turn_indexes |= turns
            if quoted is not None:
                edge.verified_turn_indexes.add(quoted)
            if not edge.evidence and item.evidence:
                edge.evidence = item.evidence.strip()

        logger.info(
            "Window %d/%d (turns %s-%s): %d node(s), %d edge(s) so far.",
            position,
            len(windows),
            window.turn_indexes[0] if window.turn_indexes else "-",
            window.turn_indexes[-1] if window.turn_indexes else "-",
            len(nodes),
            len(edges),
        )

    # A node nothing resolvable supports is a name with no evidence. It cannot
    # be retrieved through -- the turn list is what retrieval follows -- so
    # keeping it only pads the vector index with something unreachable.
    grounded = {
        identity: draft for identity, draft in nodes.items() if draft.turn_indexes
    }
    if len(grounded) < len(nodes):
        logger.info(
            "Dropped %d node(s) that cited no resolvable turn.",
            len(nodes) - len(grounded),
        )

    kept_edges = [
        edge
        for edge in edges.values()
        if by_name.get(edge.source) in grounded and by_name.get(edge.target) in grounded
    ]
    things = sum(1 for d in grounded.values() if not d.is_statement)
    logger.info(
        "Extracted %d node(s) -- %d thing(s), %d statement(s) -- and %d "
        "knowledge edge(s).",
        len(grounded),
        things,
        len(grounded) - things,
        len(kept_edges),
    )
    report_unquoted_edges(kept_edges)
    return GraphDraft(nodes=list(grounded.values()), edges=kept_edges)
