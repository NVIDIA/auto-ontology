# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entity extraction and VDB (vector-database) resolution.

Split out of clarify.py: this module owns "what entities does the question
mention, and which schema column/attribute does each one resolve to" —
running the entity-coverage LangGraph, checking VDB-hit ambiguity, and
resolving collisions where two entities land on the same column. External
knowledge (KB) coverage lives in kb_coverage.py; clarification-turn
orchestration lives in clarify.py.
"""

from __future__ import annotations

import logging
import os
import re

from langchain_core.messages import HumanMessage

from auto_ontology.retrieval.entity_coverage.graph import (
    create_graph as _create_entity_coverage_graph,
)
from auto_ontology.retrieval.entity_coverage.state import DEFAULT_MAX_DISTANCE
from auto_ontology.retrieval.data_access.semantic_search import search_semantic_index
from auto_ontology.semantic.constants import LABEL_COLUMN_ATTRIBUTE
from auto_ontology.utils.llm_invoke import get_llm_client, safe_invoke_text_nr

from .kb_coverage import _filter_covered_by_external_knowledge, _parse_kb_entries

logger = logging.getLogger(__name__)


def _resolve_clarify_max_distance() -> float:
    """Resolve the VDB distance threshold used during clarification.

    CLARIFY_MAX_DISTANCE env var, if set, wins outright (explicit float
    override). Otherwise INTERACTIVE=true selects the stricter 0.65 threshold
    tuned for the interactive clarification flow; when INTERACTIVE is unset
    or false, this falls back to the general DEFAULT_MAX_DISTANCE (0.75).
    """
    explicit = os.environ.get("CLARIFY_MAX_DISTANCE", "")
    if explicit:
        return float(explicit)
    if os.environ.get("INTERACTIVE", "").lower() in ("true", "1"):
        return 0.65
    return DEFAULT_MAX_DISTANCE


# Distance threshold for VDB resolution: entity score must be <= this value with
# no ambiguous second hit to count as "found in schema". Lower = stricter.
# See _resolve_clarify_max_distance() for the INTERACTIVE/override logic.
CLARIFY_MAX_DISTANCE: float = _resolve_clarify_max_distance()

# Max gap between an entity's best and second-best VDB hit for the pair to
# count as a genuine tie. Without this, an entity with a confident best hit
# (e.g. 0.10) and a barely-qualifying second hit (e.g. 0.64) was flagged
# "ambiguous" just as readily as a real coin-flip (e.g. 0.30 vs 0.32) — both
# independently clearing CLARIFY_MAX_DISTANCE says nothing about whether
# they're actually close to *each other*. Requiring the gap to be small too
# keeps ambiguity flagging scoped to real near-ties.
_AMBIGUITY_MAX_GAP: float = 0.25

# Compiled entity-coverage LangGraph — shared across calls, compiled once at import.
_ec_app = _create_entity_coverage_graph().compile()

_FILLER = frozenset(
    [
        # SQL aggregation / math
        "average",
        "median",
        "mean",
        "count",
        "total",
        "sum",
        "min",
        "max",
        "number",
        "value",
        "measure",
        "metric",
        "level",
        "score",
        "ratio",
        "rate",
        "index",
        "indicator",
        "standard",
        "deviation",
        "percentage",
        "column",
        # Schema-structural words — stripping these improves VDB matching
        # e.g. "condition name" → "condition", "signal type" → "signal"
        "name",
        "type",
        "id",
        "key",
        "code",
        "label",
        "category",
    ]
)

# Structural connectives — always stripped from anywhere in the phrase,
# never counted toward the filler threshold (unlike _FILLER words).
# e.g. "number of records" → ["number", "records"] before threshold check.
_CONNECTIVES = frozenset(["of", "by"])

_ARTICLES = frozenset(["a", "an", "the"])


def _normalize_entity(entity: str) -> str:
    """Strip filler/aggregation words and leading articles so VDB search targets the core domain term.

    Connectives ("of", "by") are stripped first and never count toward the filler threshold.
    Filler words are only stripped when they make up half or fewer of the remaining words —
    if every word is a filler (e.g. "score level") or strictly more than half are fillers
    (e.g. "total point count"), the phrase is kept intact so the VDB still receives a
    meaningful query.
    """
    raw = [w for w in entity.lower().split() if w not in _CONNECTIVES]
    non_filler = [w for w in raw if w not in _FILLER]
    if not non_filler or len(non_filler) < len(raw) / 2:
        tokens = raw
    else:
        tokens = non_filler
    while tokens and tokens[0] in _ARTICLES:
        tokens.pop(0)
    return " ".join(tokens)


# Generic standalone tokens that reliably produce false-positive VDB matches via
# substring coincidence (e.g. "id" → "idle power"). The new extraction prompt also
# instructs the LLM to omit these, but a runtime guard is kept as a safety net.
# Compound entities like "customer id" are multi-token and pass through normally.
_GENERIC_STANDALONE = frozenset(
    {
        "id",
        "ids",
        "key",
        "keys",
        "value",
        "values",
        "code",
        "codes",
        "type",
        "types",
    }
)


def _run_entity_coverage_pipeline(
    question: str,
    semantic_retriever: object,
    db_name: str | None,
) -> dict:
    """Invoke the entity-coverage LangGraph and return its final path_state.

    Uses the reasoning LLM (state["llm"]) for extraction per main's decision.
    Returns an empty dict on any failure so callers degrade gracefully.
    """
    try:
        llm = get_llm_client()
    except Exception as exc:
        logger.error(
            "Clarify — could not init reasoning LLM for entity coverage: %s", exc
        )
        return {}

    path_state: dict = {
        "max_distance": CLARIFY_MAX_DISTANCE,
        "return_uncovered_entities": True,
    }
    if db_name:
        path_state["target_db"] = db_name

    state = {
        "llm": llm,
        "initial_question": question,
        "messages": [HumanMessage(content=question)],
        "path_state": path_state,
        "semantic_retriever": semantic_retriever,
        "decision": "",
        "domain_rules": [],
        # data_retriever and connectors are not used by the 3-node entity-coverage
        # graph (question_extraction → retrieve_candidates → coverage_grade) but are
        # present in AgentState; we omit them and bypass _build_state intentionally.
    }
    try:
        final_state = _ec_app.invoke(state, config={"recursion_limit": 10})
        return final_state.get("path_state", {})
    except Exception as exc:
        logger.error("Clarify — entity-coverage pipeline failed: %s", exc)
        return {}


def _ambiguity_check(
    path_state: dict,
) -> set[str]:
    """Return entity strings whose column-attribute hits are ambiguous.

    An entity is ambiguous when it retrieved 2+ column-attribute hits that both
    fall within CLARIFY_MAX_DISTANCE (meaning the VDB cannot single out one
    column) AND whose scores sit within _AMBIGUITY_MAX_GAP of each other
    (meaning they're genuinely competing, not just two hits that separately
    happened to qualify). These are demoted to unresolvable even if
    CoverageGradeAgent counted them covered.
    """
    hits: list[dict] = path_state.get("retrieved_column_attributes") or []
    # Group best score and second-best score per query_entity.
    best: dict[str, float] = {}
    second: dict[str, float] = {}
    for hit in hits:
        score = hit.get("score")
        if score is None:
            continue
        score = float(score)
        if score > CLARIFY_MAX_DISTANCE:
            continue
        for entity in hit.get("query_entities") or (
            [hit["query_entity"]] if hit.get("query_entity") else []
        ):
            if entity not in best or score < best[entity]:
                second[entity] = best.get(entity, float("inf"))
                best[entity] = score
            elif entity not in second or score < second[entity]:
                second[entity] = score
    ambiguous = {
        e
        for e, s in second.items()
        if s <= CLARIFY_MAX_DISTANCE and (s - best[e]) < _AMBIGUITY_MAX_GAP
    }
    if ambiguous:
        logger.info("Clarify — ambiguous entities (2+ close VDB hits): %s", ambiguous)
    return ambiguous


_AMBIGUOUS_HITS_TOP_K = 5  # candidates kept per ambiguous entity — enough for the
# resolver LLM to see every real contender without a prompt bloated by noise hits.


def _collect_ambiguous_hits(
    semantic_retriever: object,
    db_name: str | None,
    ambiguous_entities: set[str],
) -> dict[str, list[dict]]:
    """Fresh, per-entity top-_AMBIGUOUS_HITS_TOP_K column-attribute candidates
    for each ambiguous entity, re-queried directly against the VDB.

    Deliberately does NOT reuse `_ambiguity_check`'s already-collected
    `retrieved_column_attributes` (as this used to) or apply
    CLARIFY_MAX_DISTANCE: both `_run_entity_coverage_pipeline` and
    `_ambiguity_check` cap hits at that distance, but the *correct*
    disambiguating candidate can legitimately score just past it — e.g.
    "Counsel For" at 0.671 vs. the 0.65 cutoff, for a question asking about
    "attorney" representation, while two wrong same-table candidates both
    scored under it. Filtering it out before the resolver ever sees it isn't
    a resolution failure, it's a candidate-completeness one — confirmed by
    running the same query directly and finding the right answer sitting at
    rank 3, just outside the cutoff. This is the last, no-more-chances
    decision point (see resolve_pending_ambiguities), so it's worth the extra
    VDB call per ambiguous entity (rare) to see a wider net than detection
    needed.
    """
    return {
        e: _entity_ranked_hits(e, semantic_retriever, db_name, k=_AMBIGUOUS_HITS_TOP_K)
        for e in ambiguous_entities
    }


# ── Collision resolution ────────────────────────────────────────────────────
# When two different entities' best VDB hit lands on the same underlying
# column, decide whether one is a clear winner (auto-assign) or the scores
# are too close to call (defer to the LLM). Thresholds are deliberately
# conservative — when in doubt, defer rather than silently auto-assign.
_COLLISION_WINNER_MARGIN = (
    0.04  # winner's 1st-hit score must beat the loser's by at least this
)
_COLLISION_LOSER_MAX_GAP = (
    0.08  # loser's own gap (shared hit -> its next-distinct hit) must be under this
)

# Shared with clarify.py's KB+VDB disambiguation note (which imports this
# constant rather than defining its own copy). A term whose VDB hit scores
# below this is a "close enough" schema-column match to be worth weighing
# against a KB formula at all — anything worse and the column clearly isn't
# a real alternative, no need to ask (or to bother the merged LLM call below
# with it).
_KB_VDB_DISAMBIG_THRESHOLD = 0.62

_COMPOSITE_HIT_RE = re.compile(r"\b(json|jsonb|structured)\b", re.IGNORECASE)
# "jsonb" needs its own alternative, not just "json": `\bjson\b` requires a word
# boundary right after "json", which "JSONB" never has (the "b" is a word char
# glued onto it), so a description reading "JSONB column..." — the exact phrasing
# every *_column_meaning_base.json in this dataset uses for Postgres jsonb columns —
# silently fell through to the sample-values fallback and was missed.
# "Sample values: a, b, c" with 2+ comma-separated entries — a column description
# listing multiple distinct sample values is a reliable sign of a multi-key/composite
# column regardless of how the description happens to phrase the type (some say
# "JSON object", others just "Structured ... data" — neither keyword is guaranteed).
_SAMPLE_VALUES_RE = re.compile(r"Sample values:\s*([^.]+)", re.IGNORECASE)


def _is_composite_hit(hit: dict) -> bool:
    """Best-effort check that *hit*'s underlying column is a composite/multi-key
    column (JSON or otherwise structured), i.e. two colliding terms could both
    legitimately refer to it — as different sub-keys — rather than one of them
    being a wrong match.

    Prefers structured type metadata (``data_type``) when present; falls back
    to text checks on the hit's description otherwise.

    ``data_type`` is now threaded through from the DB's ``attr.datatype``
    (dal/terms.py -> semantic/embed.py -> data_access/semantic_search.py), but
    only for rows embedded *after* that change landed — existing VDB rows
    won't carry it until the semantic index is re-embedded. Until then this
    still falls through to the text-heuristic path below for most hits. Once
    a broad re-embed has happened, the ``data_type`` branch above should be
    handling the large majority of cases and the text-heuristic fallback can
    likely be trimmed down (or dropped) — revisit then.
    """
    data_type = hit.get("data_type") or hit.get("type")
    if data_type:
        return "json" in str(data_type).lower()
    text = str(hit.get("text") or "")
    if _COMPOSITE_HIT_RE.search(text):
        return True
    sample_match = _SAMPLE_VALUES_RE.search(text)
    if sample_match:
        values = [v.strip() for v in sample_match.group(1).split(",") if v.strip()]
        return len(values) >= 2
    return False


def _shared_column_note(entities: list[str], hit: dict, fallback_id: str) -> str:
    """Format the "these terms legitimately share one composite column" note
    injected directly into SQL-gen evidence."""
    names = ", ".join(f'"{e}"' for e in entities)
    col_name = _column_name(hit, fallback_id)
    logger.info(
        "Clarify — collision resolved as shared composite column: %s -> %s",
        entities,
        col_name,
    )
    return (
        f"Note: {names} both resolve to the same column ({col_name}) "
        f"— use the correct sub-key/field for each; they are not the same value."
    )


def _column_name(hit: dict, fallback_id: str) -> str:
    """Human-readable column name from a hit's text, e.g. 'Wage Details'."""
    return (
        re.sub(r"^ColumnAttribute:\s*", "", str(hit.get("text") or ""))
        .split(".")[0]
        .strip()
        or fallback_id
    )


def _distinct_columns_note(pairs: list[tuple[str, dict, str]]) -> str:
    """Format a "these terms resolve to these (different) columns, confirmed by
    disambiguation" note for entities that collided but were resolved to
    distinct columns (score-margin auto-resolve or disambiguation LLM split).

    Plain factual mapping, deliberately not phrased as a warning — that's
    reserved for :func:`_shared_column_note`'s same-column case, where the
    risk (conflating two different values) is real. Here the two terms were
    already determined to mean different things; the note just saves SQL-gen
    from re-deriving via its own, independent VDB search a resolution that
    plain vector search was already shown to get ambiguous once.

    *pairs* is a list of (entity, hit, fallback_id) — one entry per resolved
    entity, kept together as a single note when they came from the same
    collision so the mapping reads as one fact rather than scattered lines.
    """
    mappings = "; ".join(
        f'"{entity}" resolves to {_column_name(hit, fallback_id)}'
        for entity, hit, fallback_id in pairs
    )
    return f"Note (confirmed by disambiguation, not plain vector search): {mappings}."


def _entity_ranked_hits(
    entity: str, semantic_retriever: object, db_name: str | None, k: int = 5
) -> list[dict]:
    """Fresh, per-entity ranked VDB candidates for *entity* (best first).

    Deliberately re-queries the VDB directly rather than reusing the pipeline's
    already-retrieved col_hits: those are deduped globally by column id across
    all entities in the question (keeping only the single lowest score per
    id), so two colliding entities show up with an *identical* shared score —
    the exact per-entity margin this resolution needs is discarded by that
    dedup. Re-querying is only done for entities that actually collide, so the
    extra VDB calls are rare.
    """
    try:
        hits = search_semantic_index(
            semantic_retriever,
            entity,
            label_filter=[LABEL_COLUMN_ATTRIBUTE],
            per_label_k=k,
            database_name=db_name,
        )
    except Exception:
        logger.warning(
            "Clarify — collision re-query failed for %r", entity, exc_info=True
        )
        return []
    return sorted(hits, key=lambda h: float(h.get("score") or float("inf")))


_COLLISION_LLM_PROMPT = """\
Two or more terms extracted from a question both resolved, via vector search, to the \
SAME database column, which indicated potentially at least one of them is wrong. Decide \
which column each term actually refers to.

Some terms may also list a candidate with id "KB_FORMULA" — this is NOT a database \
column. It means the term already has a known calculation formula or description (shown in that \
candidate's own text) and does not need to be computed from any single raw column at \
all. Pick "KB_FORMULA" for a term when its own formula or descriptionis what actually answers the \
question, rather than forcing it onto a column that only superficially matches the term's \
name.

Question: {question}

Relevant knowledge: {relevant_knowledge}

Terms and their top candidate columns (best match first):
{candidates_block}

For each term, output exactly one line:
<term>: <chosen candidate id>

Pick the candidate id that best matches what the term refers to — it does not have to be \
the first-listed one. If a term genuinely doesn't match any candidate shown, output:
<term>: NONE
"""


# ── Ambiguous-entity resolution ─────────────────────────────────────────────
# Same call shape as collision resolution above (short list of candidates per
# term -> a fast LLM picks the id) but a different trigger and a different
# moment in the pipeline: collision resolution runs mid-dialogue, on two
# entities sharing one hit, while there may still be turns left to ask about
# it. This runs exactly once, at the decision to proceed to SQL generation —
# the last point before the question is final and no further clarifying
# question is possible — on entities where one term has 2+ column-attribute
# hits close enough to compete (_ambiguity_check). It replaces the vague
# "choose whichever fits, not both" disambiguation note (clarify.py) with an
# actual decision where it can confidently make one.

_AMBIGUITY_RESOLUTION_PROMPT = """\
A term extracted from the question matched 2+ database columns with close enough vector-search \
scores that neither could be confidently picked as the match based on score alone. \
Using all context below, decide which candidate column each term refers to. \

Question: {question}

Terms and their close-scoring candidate columns (best match first):
{candidates_block}

Potentially relevant knowledge: {relevant_knowledge}

For each term, output exactly one line:
<term>: <chosen candidate id>

Pick the candidate id that best fits what the term means in the context of this specific \
question — it does not have to be the first-listed one, or the one whose name most literally \
matches the term's wording. If a term has no likely matches, output:\
<term>: NONE
"""


# Same derivation as evidence.py's _MAX_EVIDENCE_LINE_CHARS: comfortably above
# the longest real column description in the dataset, so full definitions
# (including "Sample values: ..." tails _is_composite_hit relies on) reach the
# LLM intact instead of being cut off mid-description. Previously collision
# resolution used its own separate, much shorter 150-char cutoff on the
# (unfounded) assumption that it was a "shorter use case" — measured against
# real embedding text this was wrong even for a plain column (a description
# alone commonly runs 200-300 chars, before "Sample values:" ever starts), so
# both mechanisms now share this one budget instead of drifting independently.
_CANDIDATE_HIT_MAX_CHARS = 600

# JSON-typed columns need more room than that: their disambiguating content is
# a whole list of key+example entries (e.g. "Tx_Adh [e.g. 'High'], Func_Impv
# [e.g. 'Moderate'], ..."), not one short value — measured at ~850 chars for
# an 11-key column already close to _CANDIDATE_HIT_MAX_CHARS, and up to 20
# keys are possible (see visit_enter.py's per-column cap), so this is sized
# for a full 20-key column rather than the single example seen so far.
_CANDIDATE_HIT_MAX_CHARS_JSON = 1600

# A "KB_FORMULA" candidate (see _resolve_collisions' merged KB-formula-vs-
# column decision) is a different animal again: a formula/definition, not a
# column description or a key list. Truncating a description loses color;
# truncating a formula mid-expression can look complete while being silently
# wrong, which is worse than showing nothing. Measured real KB entry text
# (name + description + definition) across 5 databases: max 643 chars
# (archeology_scan); this candidate's own ~70-char prefix pushes the observed
# worst case to ~710, so this leaves real margin rather than barely fitting
# the one case measured.
_CANDIDATE_HIT_MAX_CHARS_KB_FORMULA = 1000


def _candidate_text_budget(hit: dict) -> int:
    """Text-length budget for one candidate line — KB_FORMULA and JSON-typed
    columns each get more room than a plain column, for different reasons
    (see the constants above)."""
    if hit.get("id") == "KB_FORMULA":
        return _CANDIDATE_HIT_MAX_CHARS_KB_FORMULA
    data_type = hit.get("data_type") or hit.get("type") or ""
    return (
        _CANDIDATE_HIT_MAX_CHARS_JSON
        if "json" in str(data_type).lower()
        else _CANDIDATE_HIT_MAX_CHARS
    )


def _format_candidate_hit(hit: dict) -> str:
    """One candidate line: id, data type (if known), and its full description."""
    data_type = hit.get("data_type") or hit.get("type")
    type_tag = f" [{data_type}]" if data_type else ""
    text = str(hit.get("text") or "")[: _candidate_text_budget(hit)]
    return f"  id={hit.get('id')}{type_tag}: {text}"


_CHOSEN_ID_PREFIX_RE = re.compile(r"^\s*id\s*=\s*", re.IGNORECASE)


def _clean_chosen_id(chosen: str) -> str:
    """Normalize a resolver LLM's raw answer for a term into a bare candidate id.

    The prompts ask for a bare id (`<term>: <id>`), but the LLM sometimes
    echoes the `id=` prefix from the candidate list itself (`<term>: id=<id>`)
    despite that — observed live, not hypothetical (see exchange_traded_funds_10
    run). Stripping it here is more reliable than depending on prompt wording
    alone to prevent the drift; a whole batch of otherwise-correct answers
    shouldn't be discarded over one stray prefix.
    """
    chosen = _CHOSEN_ID_PREFIX_RE.sub("", chosen.strip())
    return chosen.strip().strip("'\"")


def resolve_ambiguous_entities(
    question: str,
    relevant_knowledge: str,
    entity_candidates: dict[str, list[dict]],
) -> dict[str, str] | None:
    """Ask a fast LLM to pick the correct candidate id for each ambiguous entity.

    Same fail-safe contract as :func:`_llm_disambiguate_collision`: returns
    ``None`` (never a guess) on any parse failure or missing entity, so the
    caller can fall back to leaving the ambiguity unresolved rather than
    silently injecting a possibly-wrong mapping.
    """
    blocks = []
    for entity, hits in entity_candidates.items():
        lines = "\n".join(_format_candidate_hit(h) for h in hits)
        blocks.append(f'"{entity}":\n{lines}')
    prompt = _AMBIGUITY_RESOLUTION_PROMPT.format(
        question=question,
        relevant_knowledge=relevant_knowledge or "(none)",
        candidates_block="\n\n".join(blocks),
    )
    try:
        response = safe_invoke_text_nr(prompt).strip()
    except Exception:
        logger.warning("Clarify — ambiguity-resolution LLM call failed", exc_info=True)
        return None

    valid_ids = {str(h.get("id")) for hits in entity_candidates.values() for h in hits}
    result: dict[str, str] = {}
    for line in response.splitlines():
        if ":" not in line:
            continue
        term, _, chosen = line.partition(":")
        term = term.strip().strip('"')
        chosen = _clean_chosen_id(chosen)
        if term in entity_candidates and (
            chosen in valid_ids or chosen.upper() == "NONE"
        ):
            result[term] = chosen
    if len(result) != len(entity_candidates):
        logger.warning(
            "Clarify — ambiguity-resolution LLM response incomplete/unparseable: %r",
            response[:300],
        )
        return None
    return result


def resolve_pending_ambiguities(
    question: str,
    relevant_knowledge: str,
    ambiguous_hits: dict[str, list[dict]],
) -> list[str]:
    """Resolve every still-open ambiguous entity in *ambiguous_hits* and return
    evidence notes for the ones it could confidently decide.

    Called exactly once, at the decision to proceed to SQL generation (see
    coordinator.py) — *ambiguous_hits* holds whichever entities the last
    `_ambiguity_check` (against the final working_question) flagged; see
    `session._last_ambiguous_hits`'s docstring for why it's reset each turn
    rather than accumulated. Entities the LLM can't confidently resolve are
    simply skipped (no note) rather than guessed — the existing "choose
    whichever fits" disambiguation note (clarify.py) still applies as a
    fallback for those, unchanged.
    """
    if not ambiguous_hits:
        return []
    decision = resolve_ambiguous_entities(question, relevant_knowledge, ambiguous_hits)
    if decision is None:
        logger.warning(
            "Clarify — ambiguity resolution unresolved for all of: %s",
            sorted(ambiguous_hits),
        )
        return []
    by_id = {str(h.get("id")): h for hits in ambiguous_hits.values() for h in hits}
    notes = []
    for entity, chosen_id in decision.items():
        if chosen_id.upper() == "NONE" or chosen_id not in by_id:
            continue
        col_name = _column_name(by_id[chosen_id], chosen_id)
        notes.append(
            f'"{entity}" means {col_name}. Do not substitute a different column name.'
        )
    logger.info("Clarify — ambiguity resolution decided: %s", decision)
    return notes


def _llm_disambiguate_collision(
    question: str,
    relevant_knowledge: str,
    entity_candidates: dict[str, list[dict]],
) -> dict[str, str] | None:
    """Ask a fast LLM to assign each colliding entity to a distinct candidate id.

    Returns {entity: chosen_id}. Returns None on any parse failure or if any
    entity is missing from the response — callers must treat None as "could
    not resolve" and fail safe (never guess a mapping ourselves here).
    """
    blocks = []
    for entity, hits in entity_candidates.items():
        # Shares _format_candidate_hit's budget with resolve_ambiguous_entities
        # (see _CANDIDATE_HIT_MAX_CHARS / _CANDIDATE_HIT_MAX_CHARS_JSON) rather
        # than the previous separate, unjustified 150-char cutoff — collision
        # resolution is an equally consequential decision, not a "shorter use
        # case", and needs the same "Sample values: ..." tail to disambiguate.
        lines = "\n".join(_format_candidate_hit(h) for h in hits)
        blocks.append(f'"{entity}":\n{lines}')
    prompt = _COLLISION_LLM_PROMPT.format(
        question=question,
        relevant_knowledge=relevant_knowledge or "(none)",
        candidates_block="\n\n".join(blocks),
    )
    try:
        response = safe_invoke_text_nr(prompt).strip()
    except Exception:
        logger.warning("Clarify — collision LLM call failed", exc_info=True)
        return None

    valid_ids = {str(h.get("id")) for hits in entity_candidates.values() for h in hits}
    result: dict[str, str] = {}
    for line in response.splitlines():
        if ":" not in line:
            continue
        term, _, chosen = line.partition(":")
        term = term.strip().strip('"')
        chosen = _clean_chosen_id(chosen)
        if term in entity_candidates and (
            chosen in valid_ids or chosen.upper() == "NONE"
        ):
            result[term] = chosen
    if len(result) != len(entity_candidates):
        logger.warning(
            "Clarify — collision LLM response incomplete/unparseable: %r",
            response[:300],
        )
        return None
    return result


def _resolve_collisions(
    question: str,
    relevant_knowledge_text: str,
    best_hit_per_entity: dict[str, dict],
    semantic_retriever: object,
    db_name: str | None,
    kb_covered_norms: set[str] | None = None,
    hit_verdicts: dict[str, bool] | None = None,
    kb_text_by_norm: dict[str, tuple[str, str]] | None = None,
    adjudicated_norms_out: set[str] | None = None,
) -> list[str]:
    """Resolve entities whose best VDB hit collides with another entity's, in place.

    Mutates *best_hit_per_entity* so every entity ends up mapped to a distinct
    column, or is explicitly confirmed as legitimately sharing one (the JSON
    case). Returns notes to inject directly into SQL-gen evidence (bypassing
    evidence-gen's LLM, which isn't reliable about preserving instructions
    passed through it) — both "shared column" notes (JSON case) and "resolved
    to these distinct columns" notes, so SQL-gen's own independent VDB search
    doesn't have to re-derive (and risk re-getting-wrong) a resolution that
    plain vector search was already shown to be ambiguous about once.

    *hit_verdicts* (mutated in place, keyed by column id) persists across
    calls — pass ``session._collision_hit_verdicts`` to make decisions durable
    across turns. Once a column id has a verdict (True = trust as a
    legitimate multi-entity target, False = don't), a later collision on the
    *same* id skips straight to that verdict instead of re-deriving it — no
    repeat LLM call, and immune to entity-name drift across turns (e.g.
    "prevailing wage" vs "prevailing_wage level" colliding on the same id in
    different turns), since the cache key is the stable column id, not the
    entity string.

    *kb_text_by_norm* (normalized term -> (KB entry name, KB formula text))
    lets Step 0 offer the KB formula itself as a candidate to the
    disambiguation LLM, instead of always blind-dropping a KB-covered
    entity's column claim — the name travels with the text so the resulting
    note can identify which KB entry won, not just say "a KB formula".
    Optional — pass ``None`` to keep the old blind-drop behavior everywhere
    (e.g. from a caller that hasn't computed KB text per term).

    *adjudicated_norms_out* (mutated in place, a set of normalized terms) is
    populated with every term Step 0 routed to the merged LLM decision (with
    or without a competitive score — see Step 0 below), so a caller like
    clarify.py's KB+VDB disambiguation note can skip re-deciding the same term
    independently later, off a possibly-stale VDB score, and contradicting
    what the merge already adjudicated.

    Resolution order, most confident/cheapest first:
      0. Either colliding entity is already KB-covered (has a formula from
         external knowledge, e.g. "Aggressive Trading Intensity"). If its own
         VDB hit isn't a competitive match (score >= _KB_VDB_DISAMBIG_THRESHOLD),
         it doesn't need a raw column identity at all; drop it from
         best_hit_per_entity rather than let it win or get auto-assigned one —
         it only showed up as a VDB "entity" because the working question
         repeats its name from a clarification answer, not because it's a
         schema concept. If its VDB hit *is* competitive, though, the column
         is a real enough alternative that guessing either way (auto-drop, or
         clarify.py's separate KB+VDB note silently assuming both are valid)
         risks the two signals disagreeing with no one having actually
         weighed them together — route it to the same LLM disambiguation
         call as any other colliding entity, with the KB formula added as one
         more candidate for it to choose (see the "KB_FORMULA" candidate
         id below), which also opts it out of the score-margin auto-resolve
         in step 3 for the same reason (that heuristic doesn't know a KB
         formula is on the table).
      1. Otherwise, the shared id already has a cached verdict -> apply it
         directly, no re-derivation, no LLM call.
      2. Otherwise, shared hit is a JSON column -> assume both terms correctly
         share it (they likely need different sub-keys within it); note it,
         don't reassign, cache verdict True.
      3. Otherwise, if one term's 1st-hit score beats the other's by >= 0.04
         AND the loser's own gap to its next-distinct candidate is < 0.08:
         auto-assign the loser to that next-distinct candidate; note the
         distinct-column resolution.
      4. Otherwise: ask a fast LLM to pick, given both terms' top candidates.
      5. If the LLM call fails to parse: drop the lower-confidence entity from
         best_hit_per_entity entirely (treat as VDB-unresolved) rather than guess.
    """
    notes: list[str] = []
    kb_covered_norms = kb_covered_norms or set()
    hit_verdicts = {} if hit_verdicts is None else hit_verdicts
    kb_text_by_norm = kb_text_by_norm or {}

    id_to_entities: dict[str, list[str]] = {}
    for entity, hit in best_hit_per_entity.items():
        id_to_entities.setdefault(str(hit.get("id") or ""), []).append(entity)
    collisions = {hid: ents for hid, ents in id_to_entities.items() if len(ents) > 1}
    if not collisions:
        return notes

    for shared_id, entities in collisions.items():
        # Step 0: a KB-covered entity's column claim is either dropped (not a
        # competitive score — the column clearly isn't a real alternative) or
        # routed to the merged LLM decision below (competitive score — a real
        # "KB formula vs. this column" question, see the docstring). Entities
        # in kb_formula_candidates stay in `entities` and in
        # best_hit_per_entity; only the genuinely non-competitive ones get
        # dropped here.
        kb_covered_here = [
            e
            for e in entities
            if (_normalize_entity(e) or e.lower().strip()) in kb_covered_norms
        ]
        kb_formula_candidates: dict[str, tuple[str, str]] = {}  # entity -> (name, text)
        for e in kb_covered_here:
            norm = _normalize_entity(e) or e.lower().strip()
            kb_entry = kb_text_by_norm.get(norm)
            score = best_hit_per_entity[e].get("score")
            score = float(score) if score is not None else float("inf")
            if kb_entry and score < _KB_VDB_DISAMBIG_THRESHOLD:
                kb_formula_candidates[e] = kb_entry
                if adjudicated_norms_out is not None:
                    adjudicated_norms_out.add(norm)
                continue
            logger.info(
                "Clarify — collision: %r is already KB-covered (has a formula), "
                "dropping its VDB column claim to %r instead of resolving it",
                e,
                shared_id,
            )
            best_hit_per_entity.pop(e, None)
        entities = [
            e
            for e in entities
            if e not in kb_covered_here or e in kb_formula_candidates
        ]
        if len(entities) < 2:
            # Not a real multi-entity collision anymore. A lone competitive
            # kb_formula_candidate left here (never popped, still in
            # best_hit_per_entity) is a single-term "KB formula vs. this one
            # column" question — exactly what clarify.py's standalone KB+VDB
            # disambiguation note already handles; leave it for that
            # mechanism instead of duplicating the decision here.
            continue  # no real collision left once KB-covered terms are removed
        shared_hit = best_hit_per_entity[entities[0]]

        if kb_formula_candidates:
            # At least one entity here has a competitive KB formula as an
            # alternative to the shared column. None of steps 1-3 below know
            # that option exists — the cached verdict (step 1) answers a
            # different question ("is this column safe as a shared target",
            # not "should this entity use it or its KB formula"), the
            # composite/JSON shortcut (step 2) would silently assume sharing
            # is fine without ever weighing the KB formula, and the
            # score-margin heuristic (step 3) can't see the KB option either.
            # So skip straight to the LLM call with every entity in the
            # group as a participant, KB formula injected as a candidate for
            # whichever entities have one.
            fresh_hits = {
                e: _entity_ranked_hits(e, semantic_retriever, db_name) for e in entities
            }
            entity_candidates: dict[str, list[dict]] = {}
            for e in entities:
                cands = list(fresh_hits[e][:3])
                if e in kb_formula_candidates:
                    kb_name, kb_text = kb_formula_candidates[e]
                    cands = [
                        {
                            "id": "KB_FORMULA",
                            "data_type": None,
                            "text": (
                                f"Use the KB-defined formula {kb_name!r} for this "
                                f"term instead of a schema column: {kb_text}"
                            ),
                        }
                    ] + cands
                entity_candidates[e] = cands
            decision = _llm_disambiguate_collision(
                question, relevant_knowledge_text, entity_candidates
            )
            if decision is None:
                for e in entities:
                    logger.warning(
                        "Clarify — collision (KB-formula variant) unresolved for "
                        "%r; dropping VDB hit",
                        e,
                    )
                    best_hit_per_entity.pop(e, None)
                hit_verdicts[shared_id] = False
                continue
            by_id = {
                str(h.get("id")): h for hits in entity_candidates.values() for h in hits
            }
            for entity, chosen_id in decision.items():
                if chosen_id == "KB_FORMULA":
                    kb_name = kb_formula_candidates.get(entity, ("", ""))[0]
                    logger.info(
                        "Clarify — collision LLM-resolved: %r uses its KB formula "
                        "%r, not a schema column",
                        entity,
                        kb_name,
                    )
                    best_hit_per_entity.pop(entity, None)
                    notes.append(
                        f'Note (confirmed by disambiguation): "{entity}" is defined '
                        f"by its KB formula ({kb_name}), not a direct schema column."
                    )
                elif chosen_id.upper() == "NONE" or chosen_id not in by_id:
                    best_hit_per_entity.pop(entity, None)
                else:
                    best_hit_per_entity[entity] = by_id[chosen_id]
            logger.info(
                "Clarify — collision LLM-resolved (KB-formula variant): %s", decision
            )
            # This shared_id's fate was decided alongside a KB-formula
            # question specific to these entities, not a general "is this
            # column a legitimate shared target" fact — don't cache it for
            # reuse by an unrelated future collision on the same id.
            continue

        # Step 1: a past collision (this turn or an earlier one) already
        # settled whether this exact column id is safe to trust as shared.
        cached = hit_verdicts.get(shared_id)
        if cached is True:
            logger.info(
                "Clarify — collision on %r skipped (cached trusted verdict): %s",
                shared_id,
                entities,
            )
            # Call for its logging side effect only — don't append the
            # returned note to Evidence. SQL-gen's system prompt treats the
            # entire Evidence section as authoritative ground truth that
            # overrides its own interpretation, same trust level as a
            # verified fact, even though this is an assumed (not
            # LLM-confirmed) sharing. Caching to skip re-derivation is kept;
            # only feeding the guess into Evidence as fact is removed.
            _shared_column_note(entities, shared_hit, shared_id)
            continue
        if cached is False:
            # Same bypass as Step 0's KB-covered exclusion: this id's fate is
            # already known, so don't run any collision machinery on it again
            # — no fresh score lookups, no heuristic, no LLM. Just drop these
            # entities as unresolved, same as the LLM-parse-failure fail-safe.
            logger.info(
                "Clarify — collision on %r skipped (cached distrust verdict) — "
                "dropping as unresolved, no re-check: %s",
                shared_id,
                entities,
            )
            for e in entities:
                best_hit_per_entity.pop(e, None)
            continue

        # cached is always None here — True/False both returned above already.
        if _is_composite_hit(shared_hit):
            hit_verdicts[shared_id] = True
            # Called for its logging side effect only — see the Step-1
            # cache-reuse branch above for why the note itself isn't
            # appended to Evidence: this is a blind assumption (JSON-typed
            # shared hit implies sharing is fine), never confirmed by an
            # LLM, and Evidence treats whatever lands in it as ground truth
            # regardless.
            _shared_column_note(entities, shared_hit, shared_id)
            continue

        # Fresh per-entity queries — see _entity_ranked_hits for why col_hits can't be reused.
        fresh_hits = {
            e: _entity_ranked_hits(e, semantic_retriever, db_name) for e in entities
        }

        def _score_for_shared(e: str) -> float:
            hit = next(
                (h for h in fresh_hits[e] if str(h.get("id") or "") == shared_id), None
            )
            return (
                float(hit["score"])
                if hit
                else float(best_hit_per_entity[e].get("score") or float("inf"))
            )

        ranked = sorted(entities, key=_score_for_shared)
        winner = ranked[0]
        winner_score = _score_for_shared(winner)

        needs_llm: list[str] = []
        for loser in ranked[1:]:
            loser_score = _score_for_shared(loser)
            next_distinct = next(
                (h for h in fresh_hits[loser] if str(h.get("id") or "") != shared_id),
                None,
            )
            margin_ok = (loser_score - winner_score) >= _COLLISION_WINNER_MARGIN
            gap_ok = (
                next_distinct is not None
                and (float(next_distinct["score"]) - loser_score)
                < _COLLISION_LOSER_MAX_GAP
            )
            if margin_ok and gap_ok:
                logger.info(
                    "Clarify — collision auto-resolved: %r kept %r (%.3f); "
                    "%r reassigned to %r (%.3f)",
                    winner,
                    shared_id,
                    winner_score,
                    loser,
                    next_distinct.get("id"),
                    float(next_distinct["score"]),
                )
                best_hit_per_entity[loser] = next_distinct
                notes.append(
                    _distinct_columns_note(
                        [
                            (winner, shared_hit, shared_id),
                            (loser, next_distinct, str(next_distinct.get("id") or "")),
                        ]
                    )
                )
            else:
                needs_llm.append(loser)

        if not needs_llm:
            hit_verdicts[shared_id] = False
            continue

        entity_candidates = {e: fresh_hits[e][:3] for e in [winner, *needs_llm]}
        decision = _llm_disambiguate_collision(
            question, relevant_knowledge_text, entity_candidates
        )
        if decision is None:
            # Fail safe: don't guess. Drop the lower-confidence entities so they
            # fall through the normal "VDB-unresolved" path instead of silently
            # keeping a possibly-wrong shared mapping.
            for loser in needs_llm:
                logger.warning(
                    "Clarify — collision unresolved for %r; dropping VDB hit", loser
                )
                best_hit_per_entity.pop(loser, None)
            hit_verdicts[shared_id] = False
            continue

        by_id = {
            str(h.get("id")): h for hits in entity_candidates.values() for h in hits
        }
        for entity, chosen_id in decision.items():
            if chosen_id.upper() == "NONE" or chosen_id not in by_id:
                best_hit_per_entity.pop(entity, None)
            else:
                best_hit_per_entity[entity] = by_id[chosen_id]
        logger.info("Clarify — collision LLM-resolved: %s", decision)

        # The LLM may itself decide two entities genuinely share a column (assign
        # them the same id) rather than picking distinct ones. We don't second-guess
        # that decision — it's trusted as-is, same as every other id it chose. The
        # only thing we add on top: if that shared id is detectably composite,
        # attach the "use distinct sub-keys" note so evidence-gen doesn't treat it
        # as one plain value for both terms. If it's not detectably composite, we
        # still accept the LLM's answer — we just don't have a note to add.
        new_id_to_entities: dict[str, list[str]] = {}
        for entity, chosen_id in decision.items():
            if chosen_id.upper() != "NONE" and chosen_id in by_id:
                new_id_to_entities.setdefault(chosen_id, []).append(entity)

        # Entities the LLM split onto distinct ids: note the mapping, and mark
        # the original shared_id as distrusted (this LLM call is the strongest
        # evidence we have that shared_id itself isn't a shared target) — unless
        # the LLM converged entities right back onto shared_id itself, in which
        # case it's confirmed trusted, not distrusted. Checked by id, not just
        # "did any convergence happen anywhere" — convergence onto a *different*
        # id says nothing about shared_id's own trustworthiness.
        converged_entities = {
            e for ents in new_id_to_entities.values() if len(ents) >= 2 for e in ents
        }
        split_pairs = [
            (entity, by_id[chosen_id], chosen_id)
            for entity, chosen_id in decision.items()
            if chosen_id.upper() != "NONE"
            and chosen_id in by_id
            and entity not in converged_entities
        ]
        if split_pairs:
            notes.append(_distinct_columns_note(split_pairs))
        hit_verdicts[shared_id] = len(new_id_to_entities.get(shared_id, [])) >= 2

        for new_id, ents in new_id_to_entities.items():
            # Same Step-0 exclusion as the pre-LLM collision above: a KB-covered
            # entity doesn't need a column identity at all, regardless of what
            # the LLM assigned it.
            kb_covered_here = [
                e
                for e in ents
                if (_normalize_entity(e) or e.lower().strip()) in kb_covered_norms
            ]
            for e in kb_covered_here:
                logger.info(
                    "Clarify — collision (post-LLM): %r is already KB-covered, "
                    "dropping its VDB column claim to %r instead of resolving it",
                    e,
                    new_id,
                )
                best_hit_per_entity.pop(e, None)
            ents = [e for e in ents if e not in kb_covered_here]
            if len(ents) < 2:
                continue
            hit_verdicts[new_id] = True
            if _is_composite_hit(by_id[new_id]):
                # Called for its logging side effect only — see the other
                # two _shared_column_note call sites for why the note isn't
                # appended to Evidence. This one has slightly more backing
                # (an LLM actually converged these entities onto new_id, not
                # a blind assumption), but it's still the same claim shape
                # ("these share a JSON column") landing in the same
                # over-trusted Evidence channel. Caching is kept regardless.
                _shared_column_note(ents, by_id[new_id], new_id)

    return notes


def _find_unresolvable_entities(
    question: str,
    semantic_retriever: object,
    db_name: str | None,
    formatted_kb: str = "",
    children_map: dict[str, list[str]] | None = None,
    hit_verdicts: dict[str, bool] | None = None,
    ambiguous_hits_out: dict[str, list[dict]] | None = None,
    adjudicated_norms_out: set[str] | None = None,
) -> tuple[
    list[tuple[str, str | None]],
    list[tuple[str, str, float, str]],
    str,
    set[str],
    dict[str, list[str]],
    set[str],
    list[str],
]:
    """Return (unresolvable_entities, resolved_hits, relevant_knowledge_text, all_norms,
    entry_to_original_terms, vdb_only_norms, json_shared_notes).

    Flow:
      1. Entity-coverage pipeline (reasoning LLM): extracts entities + runs VDB for all
         of them against column attributes, SQL attributes, and custom analyses.
      2. Ambiguity check: any entity with 2+ column-attribute hits within CLARIFY_MAX_DISTANCE
         is demoted to unresolvable regardless of coverage grade.
      3. KB check on ALL extracted entities (not just VDB-uncovered): populates
         relevant_knowledge_text for the prompt and identifies KB-covered entities.
      3b. Collision resolution: entities whose best hit collided with another
          entity's are auto-resolved (JSON-shared / margin-based) or sent to a
          small LLM disambiguation call — see _resolve_collisions.
      4. Final unresolvable = (VDB-uncovered ∪ ambiguous) − KB-covered.

    resolved_hits contains (entity, hit_text, score, hit_id) for entities cleanly resolved
    by VDB (score <= CLARIFY_MAX_DISTANCE, unambiguous, post-collision-resolution); the
    caller uses score <= 0.63 for evidence generation. hit_id is the underlying attribute
    node's ID — a stable identity check for "do two terms resolve to the same column",
    since hit_text is only a display label and isn't guaranteed unique or consistently
    formatted across hits.
    entry_to_original_terms maps each confirmed KB entry name
    to the original natural-language terms that matched it (for cumulative_grounded_knowledge).
    json_shared_notes are "these terms share a JSON column, use distinct sub-keys" notes
    to inject verbatim into SQL-gen evidence — see _resolve_collisions.

    When *ambiguous_hits_out* is given, it is reset to exactly this turn's
    ambiguous entities and their candidate hits (Step 2) — not merged with
    whatever it held before. Deliberately last-call-wins, not accumulated
    across turns like the KB/VDB "_ever_*" bookkeeping elsewhere in this
    module: an entity ambiguous mid-dialogue may no longer even appear in a
    later, re-phrased working_question, and re-flagging it from stale hits
    would resolve a tie against a question that's no longer the one being
    asked. Callers pass a dict that survives across calls only so the very
    last call's result (this function is re-run every turn on the latest
    working_question) is what's left in it by the time resolve_pending_ambiguities
    reads it, right before SQL generation — see coordinator.py.

    *adjudicated_norms_out*, if given, is populated (not cleared — accumulate
    across calls, same as the *_ever_* pattern) with every normalized term
    _resolve_collisions routed to its merged KB-formula-vs-column LLM
    decision. Pass ``session._kb_vdb_adjudicated_norms`` so clarify.py's
    standalone KB+VDB disambiguation note can skip a term already decided
    here instead of re-deciding it off a possibly-stale VDB score.
    """
    if ambiguous_hits_out is not None:
        ambiguous_hits_out.clear()
    if semantic_retriever is None:
        return [], [], "", set(), {}, set(), []

    # --- Step 1: run entity-coverage pipeline ---
    ec_path_state = _run_entity_coverage_pipeline(question, semantic_retriever, db_name)
    if not ec_path_state:
        return [], [], "", set(), {}, set(), []

    raw_entities: list[str] = list(ec_path_state.get("entities") or [])
    if not raw_entities:
        return [], [], "", set(), {}, set(), []

    # Normalize and deduplicate for consistent downstream handling.
    # "median signal quality" and "signal quality" both → "signal quality" (one entry).
    norm_to_original: dict[str, str] = {}
    for entity in raw_entities:
        norm = _normalize_entity(entity)
        if norm and norm not in norm_to_original:
            norm_to_original[norm] = entity
    # Drop entities whose normalized form is a strict substring of another in the batch.
    # e.g. "condition" ⊂ "atmospheric conditions" → drop; "signal dynamics" ⊄ "signal quality" → keep both
    all_norms = set(norm_to_original.keys())
    search_norms = {
        e for e in all_norms if not any(e != o and e in o for o in all_norms)
    }
    logger.info("Clarify — extracted entities (normalized): %s", sorted(search_norms))

    # Strip generic standalone tokens — the prompt already excludes them but LLMs
    # occasionally emit them; a second VDB hit on "id" or "type" would be misleading.
    generic_skipped = search_norms & _GENERIC_STANDALONE
    if generic_skipped:
        logger.info("Clarify — dropping generic standalone terms: %s", generic_skipped)
    search_norms -= generic_skipped

    # --- Step 2: ambiguity check on column-attribute hits ---
    # _ambiguity_check operates on ec_path_state's raw, pre-normalization entity
    # strings (e.g. "Price-to-Book (P/B) ratio"), but everything else in this
    # function — search_norms, needs_kb_rescue, resolved_hits — keys off the
    # normalized form (_normalize_entity strips filler words like "ratio",
    # "return"; here that's "price-to-book (p/b)"). Left unnormalized, an
    # ambiguous entity with a filler word in it never matches anything in the
    # rest of the pipeline's string space (e.g. `ambiguous_entities &
    # search_norms` in needs_kb_rescue below silently comes up empty for it).
    # Normalize before handing off to the resolver so it operates in the same
    # string space as everything else it might overlap with (KB coverage,
    # collision resolution).
    ambiguous_entities = _ambiguity_check(ec_path_state)
    if ambiguous_entities and ambiguous_hits_out is not None:
        normalized_ambiguous = {
            _normalize_entity(e) or e.lower().strip() for e in ambiguous_entities
        }
        # ambiguous_hits_out was already reset to empty above — this populates
        # it fresh with exactly this call's ambiguous entities, nothing carried
        # over from an earlier turn's (possibly stale) working_question.
        ambiguous_hits_out.update(
            _collect_ambiguous_hits(semantic_retriever, db_name, normalized_ambiguous)
        )

    # Build VDB-uncovered set: entities the pipeline marked uncovered + ambiguous ones.
    # CoverageGradeAgent nests this under path_state["final_response"], not at the
    # top level (same place "candidates" lives, read via final_response below) —
    # reading it off ec_path_state directly silently returned empty every time.
    vdb_uncovered: set[str] = set(
        (ec_path_state.get("final_response") or {}).get("uncovered_entities") or []
    )
    # Normalize uncovered_entities to match our norm keys (pipeline emits raw strings).
    vdb_uncovered_norms: set[str] = set()
    for raw in vdb_uncovered:
        norm = _normalize_entity(raw)
        if norm:
            vdb_uncovered_norms.add(norm)
        else:
            vdb_uncovered_norms.add(raw.lower().strip())
    # Merge in ambiguous entities (they came back "covered" by score but are not reliable).
    needs_kb_rescue = vdb_uncovered_norms | (ambiguous_entities & search_norms)

    # Find each entity's single best VDB hit (score <= CLARIFY_MAX_DISTANCE).
    col_hits: list[dict] = ec_path_state.get("retrieved_column_attributes") or []
    candidates_by_id: dict[str, dict] = {
        c["id"]: c
        for c in (ec_path_state.get("final_response") or {}).get("candidates", [])
        if c.get("id")
    }
    best_hit_per_entity: dict[str, dict] = {}
    for hit in col_hits:
        score = hit.get("score")
        if score is None or float(score) > CLARIFY_MAX_DISTANCE:
            continue
        for entity in hit.get("query_entities") or (
            [hit["query_entity"]] if hit.get("query_entity") else []
        ):
            if entity not in best_hit_per_entity or float(score) < float(
                best_hit_per_entity[entity].get("score", float("inf"))
            ):
                best_hit_per_entity[entity] = hit

    # --- Step 3: KB check on ALL extracted entities ---
    # Run on all search_norms (not just uncovered) so relevant_knowledge_text is complete
    # and entities explained by KB don't end up in the unresolvable list.
    # Done before collision resolution below so relevant_knowledge_text is available to
    # the LLM disambiguation fallback.
    relevant_knowledge_text = ""
    entry_to_original_terms: dict[str, list[str]] = {}
    kb_covered_norms: set[str] = set()
    if formatted_kb and search_norms:
        kb_entities = [norm_to_original.get(norm, norm) for norm in search_norms]
        orig_lower_to_norm = {
            norm_to_original.get(n, n).lower(): n for n in search_norms
        }
        covered_originals, relevant_knowledge_text, entry_to_original_terms = (
            _filter_covered_by_external_knowledge(
                kb_entities, formatted_kb, question, children_map
            )
        )
        kb_covered_norms = {
            orig_lower_to_norm.get(orig, orig) for orig in covered_originals
        }
        # Covered-entry names are already logged inside
        # _filter_covered_by_external_knowledge ("external_kb covers: ...") —
        # no need to repeat the same set here.

    # Per-term KB formula (name, text), for _resolve_collisions' merged
    # KB-formula-vs-column decision — same fuzzy entry-name lookup clarify.py's
    # own KB+VDB disambiguation note uses, kept in sync with it deliberately.
    # The name travels alongside the text so the resulting evidence note can
    # name which KB entry won, not just say "a KB formula" with nothing to
    # cross-reference it against.
    kb_text_by_norm: dict[str, tuple[str, str]] = {}
    if entry_to_original_terms:
        kb_entries_parsed = _parse_kb_entries(relevant_knowledge_text)
        for entry_name, matched_terms in entry_to_original_terms.items():
            entry_text = next(
                (
                    v
                    for k, v in kb_entries_parsed.items()
                    if k.startswith(entry_name) or entry_name.startswith(k)
                ),
                "",
            )
            if not entry_text:
                continue
            for t in matched_terms:
                norm = _normalize_entity(t) or t.lower().strip()
                kb_text_by_norm[norm] = (entry_name, entry_text)

    # Resolve any entities that collided on the same best hit, in place.
    entities_before_resolution = set(best_hit_per_entity.keys())
    json_notes = _resolve_collisions(
        question,
        relevant_knowledge_text,
        best_hit_per_entity,
        semantic_retriever,
        db_name,
        kb_covered_norms,
        hit_verdicts,
        kb_text_by_norm,
        adjudicated_norms_out,
    )

    # Reconcile needs_kb_rescue against what collision resolution actually did —
    # it was computed before _resolve_collisions ran, so on its own it doesn't
    # know an entity got cleanly resolved (and would wrongly keep it excluded
    # from resolved_hits below / stuck in the final unresolvable list), nor that
    # an entity got dropped (and would wrongly be treated as resolved with no
    # hit, silently vanishing from both resolved_hits and unresolvable instead
    # of surfacing as something that may still need a clarifying question).
    for entity in entities_before_resolution | set(best_hit_per_entity.keys()):
        norm = _normalize_entity(entity) or entity.lower().strip()
        if entity in best_hit_per_entity:
            needs_kb_rescue.discard(
                norm
            )  # now cleanly resolved — no longer ambiguous/uncovered
        else:
            needs_kb_rescue.add(
                norm
            )  # collision resolution dropped it — treat as unresolved
            # (KB-covered drops are still correctly excluded downstream via
            # final_unresolvable_norms = needs_kb_rescue - kb_covered_norms)

    # Build resolved_hits from entities cleanly covered at VDB (unambiguous, within
    # threshold, and post-collision-resolution). Use the pipeline's enriched candidates
    # (DB-resolved attribute+term names) rather than raw VDB text blobs. Fall back to
    # raw text when no enriched candidate is available.
    resolved_hits: list[tuple[str, str, float, str]] = []
    for entity, hit in best_hit_per_entity.items():
        norm = _normalize_entity(entity) or entity.lower().strip()
        if norm not in needs_kb_rescue and norm in search_norms:
            candidate = candidates_by_id.get(str(hit.get("id") or ""))
            if candidate and candidate.get("attribute"):
                term = candidate.get("term") or ""
                hit_text = (
                    f"{candidate['attribute']} ({term})"
                    if term
                    else candidate["attribute"]
                )
            else:
                hit_text = hit.get("text") or ""
            resolved_hits.append(
                (norm, hit_text, float(hit.get("score", 1.0)), str(hit.get("id") or ""))
            )

    # --- Step 4: final unresolvable = (VDB-uncovered ∪ ambiguous) − KB-covered ---
    final_unresolvable_norms = needs_kb_rescue - kb_covered_norms
    # Also mark generics as unresolvable (they were never sent to VDB).
    final_unresolvable_norms |= {_normalize_entity(e) or e for e in generic_skipped}

    # VDB-only: resolved by VDB but not covered by external KB.
    # Returned so the caller can check for missing calculation formulas.
    vdb_only_norms: set[str] = {norm for norm, *_ in resolved_hits} - kb_covered_norms

    unresolvable: list[tuple[str, str | None]] = [
        (norm, None) for norm in final_unresolvable_norms
    ]

    logger.info(
        "Clarify — unresolvable after VDB+KB: %s",
        [e for e, _ in unresolvable] or "none",
    )
    logger.info(
        "Clarify — resolved by VDB: %s",
        [(e, f"{s:.3f}") for e, _, s, *_ in resolved_hits] or "none",
    )
    return (
        unresolvable,
        resolved_hits,
        relevant_knowledge_text,
        all_norms,
        entry_to_original_terms,
        vdb_only_norms,
        json_notes,
    )
