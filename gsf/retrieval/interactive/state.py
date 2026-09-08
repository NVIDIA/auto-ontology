from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .types import InteractivePhase


@dataclass
class InteractiveSessionState:
    session_id: str
    task_id: str
    db_name: str
    db_schema: str
    external_kb: str
    original_question: str
    working_question: str
    max_clarify_turns: int = 5
    clarify_history: list[dict] = field(default_factory=list)  # [{"q": ..., "a": ...}]
    phase: InteractivePhase = InteractivePhase.ROUND1_CLARIFY
    prior_round_sql: Optional[str] = None
    prior_round_question: Optional[str] = None  # working_question at round-1 submit
    latest_feedback: Optional[str] = None  # message from the external submit service
    path_state: dict = field(default_factory=dict)  # durable across GSF calls
    _pending_question: Optional[str] = (
        None  # last AskUserAction (for apply_user_answer)
    )
    _cached_unresolvable: Optional[list] = None  # cached per working_question
    _cached_unresolvable_for: Optional[str] = None  # working_question at cache time
    _cached_resolved_hits: Optional[list] = (
        None  # VDB resolved hits for current question
    )
    _collision_resolution_notes: list[str] = field(
        default_factory=list
    )  # collision-resolution notes (KB-formula-vs-column decisions + "resolved to these distinct columns") — accumulated across turns (never replaced), injected verbatim into Evidence.
    # Renamed from _json_shared_notes: that name described the "shared JSON
    # column" note specifically, but that note is no longer emitted (it
    # asserted an unconfirmed mapping with the same authority as verified
    # Evidence — see the collision-resolution fix), so the old name was
    # actively misleading about what this list actually carries now.
    _collision_hit_verdicts: dict = field(
        default_factory=dict
    )  # column id -> True/False, persists collision-resolution verdicts across turns so a later collision on the same id (even under different entity-name phrasing) skips re-deriving it
    _cached_vdb_only_norms: set = field(
        default_factory=set
    )  # VDB-resolved but KB-uncovered norms
    # KB+VDB disambiguation bookkeeping, accumulated across turns within a phase (see
    # clarify.py) — a normalized term's KB coverage and VDB hit are re-derived fresh
    # each turn and can land on different turns, so these persist both signals instead
    # of only checking the current turn's snapshot. Reset at Phase 1→2 handoff, same
    # as cumulative_grounded_knowledge, since Phase 2 is a different question.
    _ever_kb_covered_norms: set = field(default_factory=set)
    _ever_vdb_hit_norms: dict = field(default_factory=dict)  # norm -> (col_text, score)
    _ever_term_to_kb_entry: dict = field(
        default_factory=dict
    )  # norm -> (entry_name, entry_text)
    # Normalized terms already adjudicated by collision resolution's merged
    # KB-formula-vs-schema-column decision (see _resolve_collisions Step 0).
    # _ever_vdb_hit_norms freezes a term's first-ever-seen score via setdefault
    # and the KB+VDB disambiguation note (clarify.py) re-checks every turn
    # against that frozen score — without this, a term the merge already
    # decided on could still get a second, independently-derived (and
    # potentially contradicting) note from that separate mechanism on a later
    # turn. Backstop only: the primary fix is routing the same-turn overlap
    # through one decision (the merge); this catches the cross-turn gap.
    _kb_vdb_adjudicated_norms: set = field(default_factory=set)
    _grounded_knowledge: Optional[str] = (
        None  # relevant KB text extracted during coverage check
    )
    _grounded_knowledge_for: Optional[str] = (
        None  # working_question when _grounded_knowledge was set
    )
    prior_round_grounded_knowledge: str = ""  # snapshot of cumulative_grounded_knowledge at round-1 PROCEED, carried into round 2
    cumulative_grounded_knowledge: str = ""  # union of all _grounded_knowledge values seen this phase (never replaced, only grows)
    incomplete_formula_terms: list = field(
        default_factory=list
    )  # [(term, what_is_missing)]
    # Normalized (lowercased) incomplete-formula terms the forced-question guard
    # (clarify.py's never-asked override) has already generated a question for.
    # Checked directly instead of only inferring "already asked" from lexical
    # overlap with past question text — the forced question deliberately
    # paraphrases the term into business language (see _FORCED_QUESTION_PROMPT),
    # so a term like a snake_case column name or hyphenated join phrase can
    # legitimately never appear verbatim in its own generated question, which
    # made the overlap-only check re-fire on the same term turn after turn.
    _forced_terms_asked: set[str] = field(default_factory=set)
    persistent_unresolved: list[str] = field(
        default_factory=list
    )  # terms never resolved by KB/VDB; pruned after each answered turn
    resolved_persistent: set[str] = field(
        default_factory=set
    )  # terms pruned from persistent; blocked from re-accumulation
    initial_extracted_entities: list[str] = field(
        default_factory=list
    )  # all entities extracted on the first clarify call (turn 0)
    external_kb_children_map: dict[str, list[str]] = field(
        default_factory=dict
    )  # parent entry name → [full child texts]
    # Ambiguous-entity resolution (one term, 2+ close-scoring candidate columns —
    # see entity_resolution._ambiguity_check). Accumulated across turns like the
    # _ever_* KB/VDB bookkeeping above, since an entity can be flagged ambiguous on
    # one turn and the decision to proceed can happen turns later. Resolved exactly
    # once, right before SQL generation (regardless of *why* we're proceeding — turn
    # budget exhausted or the LLM judging PROCEED — see coordinator._run_sql_generation),
    # via a dedicated non-reasoning LLM call, same shape as collision resolution.
    _last_ambiguous_hits: dict = field(
        default_factory=dict
    )  # norm -> [candidate hit dicts]
    _ambiguity_resolution_notes: list[str] = field(
        default_factory=list
    )  # injected verbatim into evidence, computed once
    _ambiguity_resolved: bool = False  # guards the once-only resolution call — an empty notes list still means "already ran"
    scalar_hint: bool = (
        False  # True when output type is detected as scalar (one-way: False→True only)
    )
    data_retriever: Any = None
    semantic_retriever: Any = None
    connectors: list = field(default_factory=list)
