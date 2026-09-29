# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Global-search orchestration: validate, query, rank."""

from __future__ import annotations

from typing import Any

from sqlalchemy import Select

from auto_ontology.catalog.constants import Labels
from auto_ontology.dal import search as search_dal
from auto_ontology.semantic.constants import LABEL_COLUMN_ATTRIBUTE, LABEL_SQL_ATTRIBUTE
from auto_ontology.server.search.constants import MIN_SEARCH_LENGTH, TEXT_MATCH_CONTAINS

_PARENT_FROM_LAST_CRUMB = {Labels.COLUMN, LABEL_COLUMN_ATTRIBUTE, LABEL_SQL_ATTRIBUTE}


class SearchValidationError(ValueError):
    """Caller sent an unsupported match option or object type."""


def resolve_object_types(objects: list[str] | None) -> set[str]:
    """Return the object types to search; empty/None means all of them."""
    allowed = set(search_dal.SEARCH_OBJECT_TYPES)
    if not objects:
        return allowed
    unknown = sorted({item for item in objects if item not in allowed})
    if unknown:
        raise SearchValidationError(
            f"Unsupported search object type(s): {', '.join(unknown)}"
        )
    return set(objects)


def rank_key(search_term: str) -> Any:
    """Sort key: name contains every token, then synonym whole-word, then shorter names.

    Matching is per-token (``created-at`` → ``created`` and ``at``), so the
    name bucket must be too: the raw query as one needle would never be a
    substring of ``created_at`` and every hit would sort by length alone.

    The middle bucket reads as "has any alias" and is not one. By the time a row
    reaches here :func:`_normalize_item` has replaced ``synonyms`` with
    :func:`_matching_synonyms` of it, so a non-empty list means an alias matched
    *this* query. That is what makes this agree with ``search._list_rank``,
    which asks Postgres for ``synonym_hit`` and not for the presence of aliases:
    ranking a Term by aliases it merely has would put every aliased Term above
    every description hit, and the cap would then drop rows the count tab still
    reports.
    """
    tokens = search_dal.search_tokens(search_term)

    def _key(item: dict[str, Any]) -> tuple[int, int, str]:
        name = (item.get("name") or "").lower()
        if tokens and all(token in name for token in tokens):
            bucket = 0
        elif item.get("synonyms"):
            bucket = 1
        else:
            bucket = 2
        return (bucket, len(name), name)

    return _key


#: The largest share of one page the synonym rescue may claim.
#:
#: The rescue evicts from the worst-ranked end first, and every swap it makes is
#: a name or description hit the user loses. Unbounded -- and the set of Terms
#: an alias reaches is unbounded, which is why the DAL is asked to stop at this
#: many -- a query matching a few hundred of them walks the page down to index 0
#: and takes the exact name match with it.
_SYNONYM_RESCUE_SHARE = 0.1


def _synonym_rescue_budget(limit: int) -> int:
    """How many alias-only Terms one page of *limit* rows may seat.

    Read twice -- once to tell the DAL how many to fetch, once to bound the
    eviction in :func:`_fit_list_limit` -- and it has to be one definition.
    Fetching more than can be seated spends a breadcrumb lookup and a
    normalisation on rows that are then thrown away; fetching fewer under-fills
    the rescue, and does it silently, because a short page and a page with
    nothing left to rescue look identical from here.
    """
    return max(1, int(limit * _SYNONYM_RESCUE_SHARE))


def _fit_list_limit(
    items: list[dict[str, Any]], *, search_term: str
) -> list[dict[str, Any]]:
    """Cap the ranked list at ``LIST_LIMIT`` without dropping synonym-only Terms.

    Fulltext already filled the 200 slots with name/description hits, so a
    Term that matched only via ``synonyms`` would otherwise be sliced away
    on the All tab even though the count tab still includes it.

    Rescuing them costs a name hit each, so it is bounded by
    :func:`_synonym_rescue_budget` -- aliases stay visible without becoming the
    whole page. The DAL is capped at the same number, so ``extras`` is normally
    already that short; the slice is what keeps the bound true of any caller,
    including one that passes no ``synonym_limit``.
    """
    key = rank_key(search_term)
    items.sort(key=key)
    limit = search_dal.LIST_LIMIT
    if len(items) <= limit:
        return items
    kept = items[:limit]
    kept_ids = {item["id"] for item in kept}
    extras = [
        item
        for item in items[limit:]
        if item.get("synonyms") and item["id"] not in kept_ids
    ]
    if not extras:
        return kept
    replaceable = [index for index, item in enumerate(kept) if not item.get("synonyms")]
    for extra in extras[: _synonym_rescue_budget(limit)]:
        if not replaceable:
            break
        kept[replaceable.pop()] = extra
    kept.sort(key=key)
    return kept


def _normalize_crumb(crumb: dict[str, Any]) -> dict[str, Any] | None:
    if not crumb.get("name"):
        return None
    out: dict[str, Any] = {
        "name": crumb.get("name"),
        "type": crumb.get("type"),
    }
    crumb_id = crumb.get("id")
    if crumb_id:
        out["id"] = crumb_id
    return out


def _matching_synonyms(row: dict[str, Any], synonym_tokens: list[str]) -> list[str]:
    out: list[str] = []
    for raw in row.get("synonyms") or []:
        if not isinstance(raw, str) or raw in out:
            continue
        if search_dal.synonym_matches_tokens(raw, synonym_tokens):
            out.append(raw)
    return out


def _normalize_item(
    row: dict[str, Any], *, synonym_tokens: list[str]
) -> dict[str, Any]:
    crumbs: list[dict[str, Any]] = []
    for crumb in row.get("breadcrumbs") or []:
        if not isinstance(crumb, dict):
            continue
        normalized = _normalize_crumb(crumb)
        if normalized is not None:
            crumbs.append(normalized)
    parent_id = None
    if crumbs and row.get("label") in _PARENT_FROM_LAST_CRUMB:
        parent_id = crumbs[-1].get("id")
    return {
        "id": row.get("id"),
        "name": row.get("name"),
        "type": row.get("label"),
        "table_type": row.get("table_type"),
        "description": row.get("description"),
        "certified": row.get("certified"),
        "parent_id": parent_id,
        "breadcrumbs": crumbs,
        "synonyms": _matching_synonyms(row, synonym_tokens),
    }


def _tokens_or_empty(search_term: str) -> list[str] | None:
    stripped = search_term.strip()
    if len(stripped) < MIN_SEARCH_LENGTH:
        return None
    return search_dal.search_tokens(stripped) or None


def _prepare_search(
    *,
    search_term: str,
    text_match_option: str,
    objects: list[str] | None,
    include_synonyms: bool,
) -> tuple[list[str], set[str], str, list[str]] | None:
    """Validate and expand a query. ``None`` means nothing searchable.

    *include_synonyms* is honoured by returning no synonym tokens rather than
    by a flag the rest of the module has to remember. That is the whole switch,
    and it works because every place aliases are consulted already treats an
    empty token list as "no alias can match": the DAL leaves ``synonym_hit``
    NULL and drops its alias-only branch (see ``_term_select`` and
    ``_synonym_term_select`` in ``auto_ontology/dal/search.py``), and here
    :func:`_matching_synonyms` reduces every hit's aliases to none -- which in
    turn puts :func:`rank_key`'s middle bucket and :func:`_fit_list_limit`'s
    rescue out of reach.

    So there is exactly one thing to get right, and no second definition of
    "synonyms are off" to fall out of step with the first.
    """
    if text_match_option != TEXT_MATCH_CONTAINS:
        raise SearchValidationError(
            f"Unsupported text_match_option {text_match_option!r}; "
            f"only {TEXT_MATCH_CONTAINS!r} is implemented"
        )
    tokens = _tokens_or_empty(search_term)
    if tokens is None:
        return None
    types = resolve_object_types(objects)
    stripped = search_term.strip()
    synonym_tokens = (
        search_dal.synonym_word_tokens(stripped) if include_synonyms else []
    )
    return tokens, types, stripped, synonym_tokens


def global_search(
    *,
    search_term: str,
    text_match_option: str,
    objects: list[str] | None,
    include_description: bool,
    include_synonyms: bool,
) -> dict[str, Any]:
    """Run the list path: fulltext + enrichment, capped and ranked."""
    prepared = _prepare_search(
        search_term=search_term,
        text_match_option=text_match_option,
        objects=objects,
        include_synonyms=include_synonyms,
    )
    if prepared is None:
        return {"data": [], "count": 0}

    tokens, types, stripped, synonym_tokens = prepared
    rows = search_dal.fetch_global_search(
        tokens,
        types,
        include_description=include_description,
        synonym_tokens=synonym_tokens,
        limit=search_dal.LIST_LIMIT,
        synonym_limit=_synonym_rescue_budget(search_dal.LIST_LIMIT),
    )
    items = [_normalize_item(row, synonym_tokens=synonym_tokens) for row in rows]
    items = _fit_list_limit(items, search_term=stripped)
    return {"data": items, "count": len(items)}


def match_selects(
    *,
    search_term: str,
    text_match_option: str,
    objects: list[str] | None,
    include_description: bool,
    include_synonyms: bool,
) -> dict[str, Select]:
    """The same match as :func:`global_search`, uncapped, as unexecuted
    statements — one ``SELECT id`` per object type.

    The path a rule replays on ingest. It goes through :func:`_prepare_search`
    like the other two, which is the whole point of it being here rather than
    a direct call into the DAL: what counts as a match -- the tokens, the
    object types, whether aliases are consulted -- is decided once, so a rule
    cannot come to label something the dialog that created it would not have
    shown.

    What it does *not* share with :func:`global_search` is the cap. That is
    deliberate and is the difference between the two callers: a person is shown
    a page and a rule labels a catalog, so stopping a rule at
    ``LIST_LIMIT`` would silently cap what it labels at the size of a page
    however far the catalog had grown. See
    :func:`auto_ontology.dal.search.matching_id_selects`.

    Raises for a term it cannot replay, where the other two return an empty
    result for one. That is not an inconsistency but the same answer read by
    a different caller: the list and the count are drawing a screen for
    somebody who has typed one character so far, while this feeds
    :func:`auto_ontology.server.rules.service.replay_search`, whose caller reads
    no targets as "this rule matches nothing now" and takes back every label it
    applied. A term that cannot be tokenized is not a rule that matches
    nothing, and quietly erasing its labels is the wrong half of that
    ambiguity to land on -- raising instead leaves them standing and puts the
    rule in the pass's failure tally, before either write phase touches it.

    A guard rather than a reachable path: the create route holds a term to
    ``MIN_SEARCH_LENGTH`` and an update cannot change it, so reaching this
    means a rule was written straight to the DAL.
    """
    prepared = _prepare_search(
        search_term=search_term,
        text_match_option=text_match_option,
        objects=objects,
        include_synonyms=include_synonyms,
    )
    if prepared is None:
        raise SearchValidationError(
            f"Search term {search_term!r} cannot be replayed; a rule's term "
            f"must be at least {MIN_SEARCH_LENGTH} characters and hold "
            f"something searchable"
        )

    tokens, types, _stripped, synonym_tokens = prepared
    return search_dal.matching_id_selects(
        tokens,
        types,
        include_description=include_description,
        synonym_tokens=synonym_tokens,
    )


def global_search_count(
    *,
    search_term: str,
    text_match_option: str,
    objects: list[str] | None,
    include_description: bool,
    include_synonyms: bool,
) -> dict[str, Any]:
    """Run the count path: same match as list, grouped by type, no cap."""
    prepared = _prepare_search(
        search_term=search_term,
        text_match_option=text_match_option,
        objects=objects,
        include_synonyms=include_synonyms,
    )
    if prepared is None:
        return {"data": {}}

    tokens, types, _stripped, synonym_tokens = prepared
    counts = search_dal.count_global_search(
        tokens,
        types,
        include_description=include_description,
        synonym_tokens=synonym_tokens,
    )
    return {"data": counts}
