# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rules: a saved global search, and the tags to apply to what it matches.

A rule spans two tables, and which one owns what is the thing to know here.
``rule`` holds the search — the term, the match option and the filters — and
``rule__tag`` holds the tags, one row per tag, with a foreign key to the tag
itself. The tag *ids* are therefore the database's business: a tag deleted from
the vocabulary takes its ``rule__tag`` rows with it by cascade, so no rule can
be read applying a tag that is no longer there.

Ids and an order are all that table keeps. A tag's name and dates belong to the
``tag`` table and are read through the join every time, so a renamed tag reads
back renamed on every rule applying it and there is no second copy of the
vocabulary to fall out of date. A caller posting whole tag objects is answered
with the tags as they *are*, not as it sent them.

:func:`create_rule` is one transaction over both tables: a rule with no tags
applies nothing, so a half-written one is worse than none at all. It refuses a
tagless rule outright, for the same reason and without waiting for a caller to
refuse it first.

Rule names are unique, folded for case and surrounding space, exactly as tag
names are -- ``uq_rule_name_lower`` in the schema, and :func:`_name_taken` in
front of it so the answer is a message rather than a constraint violation.

The read side is paged, and :func:`list_rules` and :func:`count_rules` take the
same *search* so a page and its total describe one list. There is no read of one
rule by id: a rule is only ever shown as a row of that list, and
:func:`update_rule` answers with the renamed rule so even an edit needs no
re-read.

:func:`update_rule` renames, and renames only: a rule's search and its tags are
what it *is*, and changing either would move which objects it labels while
leaving the labels it has already written behind.

:func:`delete_rule` leans on the cascades and then cleans up after them: the
labels the rule applied are taken back, and a tag those labels were the whole of
goes with them. Read its docstring, because that reach is the reason a rule is
deletable at all, and because a caller can ask for the labels to be kept
instead.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import ColumnElement, func, literal, or_, select
from sqlalchemy.exc import IntegrityError

from auto_ontology.dal import schema as s
from auto_ontology.dal.session import store, write_transaction

#: The rule ``uq_rule_name_lower`` indexes, as a comparison the DAL can run.
_FOLDED_NAME = func.lower(func.trim(s.rule.c.name))

#: Every column a rule read returns, in one place so the list, the single-rule
#: read and the rename cannot drift into answering with different shapes.
_COLUMNS = (
    s.rule.c.id,
    s.rule.c.name,
    s.rule.c.search_term,
    s.rule.c.text_match_option,
    s.rule.c.filters,
    s.rule.c.created,
    s.rule.c.modified,
    s.rule.c.created_by,
    s.rule.c.modified_by,
)


def _name_taken(name: str, *, exclude_id: str | None = None) -> bool:
    """Whether a rule already folds to *name* under ``uq_rule_name_lower``.

    *exclude_id* leaves one rule out, which is what a rename passes: a rule
    always collides with its own row, and the index would let the same write
    through for the same reason. Without it, re-typing a rule's own name -- or
    changing only its case -- would be reported as a name somebody else holds.
    """
    statement = select(s.rule.c.id).where(_FOLDED_NAME == name.strip().lower())
    if exclude_id is not None:
        statement = statement.where(s.rule.c.id != exclude_id)
    return bool(store().query_read(statement))


def _taken(name: str) -> ValueError:
    """The one wording for a name another rule holds, which a route turns into
    a 409. Raised from the check and from the index underneath it, so the two
    cannot be told apart by the message.
    """
    return ValueError(f"Rule with name {name!r} already exists")


def create_rule(
    *,
    name: str,
    search_term: str,
    text_match_option: str,
    filters: dict[str, Any],
    tags: list[str],
    created_by: str,
) -> str:
    """Store a rule and its tags, and return the new rule's id.

    *tags* is tag ids, and only ids: a tag's name and dates are the ``tag``
    table's to state, and a copy of them here would be a second version of the
    same facts with nothing keeping it current. Their order is recorded, so the
    rule reads back in the order the tags were picked.

    The id alone rather than the row: the route answers with it, and reading a
    rule back through the tag join to return what the caller just posted would
    be a second round trip for an answer it already has.

    Both inserts are one transaction. A rule labels nothing without its tags, so
    a rule row that survived a failed tag insert would be a rule that quietly
    does nothing rather than one that failed to save.

    Raises ``ValueError`` when another rule holds the name, from the check or
    from ``uq_rule_name_lower`` underneath it -- names are how the settings list
    refers to rules, and one name naming two of them makes a delete ambiguous.

    Raises ``ValueError`` for an empty *tags*, before anything is written. This
    module holds that a rule with no tags applies nothing, so the case is
    invalid here rather than only at whichever caller happens to check: without
    it the empty list compiles to ``INSERT INTO rule__tag DEFAULT VALUES`` and
    the answer is a not-null violation on a row nobody meant to write.

    Raises ``IntegrityError`` for a tag id that is not a tag. The route checks
    the ids first, for the sake of the 404 it owes and the message on it; this
    is what holds when a tag is deleted between that check and this write.
    """
    if not tags:
        raise ValueError("A rule must apply at least one tag")
    if _name_taken(name):
        raise _taken(name)

    with write_transaction():
        try:
            rows = store().query_write(
                s.rule.insert()
                .values(
                    name=name,
                    search_term=search_term,
                    text_match_option=text_match_option,
                    filters=filters,
                    created_by=created_by,
                )
                .returning(s.rule.c.id)
            )
        except IntegrityError as exc:
            # Translated only for the name rule. Everything else this statement
            # can fail on -- a tag id that is not a tag, on the insert below --
            # is a different answer the route owes, so it goes up untouched.
            if "uq_rule_name_lower" not in str(exc.orig):
                raise
            raise _taken(name) from exc
        rule_id = rows[0]["id"]

        store().query_write(
            s.rule__tag.insert().values(
                [
                    {"rule_id": rule_id, "tag_id": tag_id, "position": position}
                    for position, tag_id in enumerate(tags)
                ]
            )
        )

    return rule_id


def _tags_by_rule(rule_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    """The tags of each rule in *rule_ids*, in the order they were picked.

    One read for every rule rather than one per rule: the settings page lists
    them all, and the tags are joined to the vocabulary anyway.

    An id and a name, which is what a card renders and a client acts by, both
    read from the ``tag`` table through the join. So a renamed tag reads back
    renamed on every rule applying it, without a row of this table being
    touched.
    """
    rows = store().query_read(
        select(s.rule__tag.c.rule_id, s.tag.c.id, s.tag.c.name)
        .select_from(s.rule__tag.join(s.tag, s.tag.c.id == s.rule__tag.c.tag_id))
        .where(s.rule__tag.c.rule_id.in_(rule_ids))
        .order_by(s.rule__tag.c.rule_id, s.rule__tag.c.position)
    )

    tags: dict[str, list[dict[str, Any]]] = {rule_id: [] for rule_id in rule_ids}
    for row in rows:
        tags[row["rule_id"]].append({"id": row["id"], "name": row["name"]})
    return tags


def _matching(search: str | None) -> list[ColumnElement[bool]]:
    """The WHERE for *search*, or nothing at all when there is none.

    A rule matches on its own name **or** on the name of a tag it applies,
    because those two are what the settings page puts on a card -- a query that
    matched something invisible would hide rows for a reason nothing on screen
    explains.

    Case-insensitive substring, like the term list's ``query``. The tag half is
    an ``EXISTS`` rather than a join, so a rule applying three matching tags is
    one row rather than three.

    One function so :func:`list_rules` and :func:`count_rules` cannot come to
    disagree about what matches -- a page and a total taken from different
    filters would have the list asking for rows that are not there.
    """
    if search is None or search.strip() == "":
        return []

    pattern = f"%{search.strip()}%"
    tagged = (
        select(literal(1))
        .select_from(s.rule__tag.join(s.tag, s.tag.c.id == s.rule__tag.c.tag_id))
        .where(s.rule__tag.c.rule_id == s.rule.c.id, s.tag.c.name.ilike(pattern))
        .exists()
    )
    return [or_(s.rule.c.name.ilike(pattern), tagged)]


def list_rules(
    *, search: str | None = None, skip: int = 0, limit: int | None = None
) -> list[dict[str, Any]]:
    """One page of rules with their tags, ordered as the settings page renders.

    Sorted case-insensitively by name with the id as a tie-break, as tags are,
    so the order is total: two pages of one list cannot repeat or skip a rule
    because the database reshuffled equal names between the requests.

    *skip* and *limit* select a window of that order, and *limit* omitted
    returns every matching rule. :func:`count_rules` is the size of the whole
    match, which is what tells a caller when to stop asking.

    A rule always has at least one tag — the route refuses a create without one
    — so an empty ``tags`` means the vocabulary lost every tag the rule applied,
    not that the join missed.
    """
    statement = (
        select(*_COLUMNS)
        .where(*_matching(search))
        .order_by(func.lower(s.rule.c.name), s.rule.c.id)
        .offset(skip or None)
    )
    if limit is not None:
        statement = statement.limit(limit)

    rules = store().query_read(statement)
    if not rules:
        return []

    tags = _tags_by_rule([rule["id"] for rule in rules])
    return [{**rule, "tags": tags[rule["id"]]} for rule in rules]


def count_rules(*, search: str | None = None) -> int:
    """The unpaged size of :func:`list_rules`, from the same filter."""
    rows = store().query_read(
        select(func.count(s.rule.c.id).label("total")).where(*_matching(search))
    )
    return int(rows[0]["total"]) if rows else 0


def update_rule(
    *, rule_id: str, name: str, modified_by: str | None = None
) -> dict[str, Any] | None:
    """Rename a rule and return it, tags included. ``None`` for an unknown id.

    The name is taken as a plain argument rather than a dict of updates, as
    :func:`auto_ontology.dal.tags.update_tag` takes a tag's: it is the only part of a
    rule an edit changes. The search is not editable -- re-pointing a rule at
    another search would silently move which objects it labels, and the labels
    it has already written would be left behind explaining nothing -- and
    neither are the tags, for that same reason from the other side.

    *name* is stored as given, as on :func:`create_rule`; the caller trims it.

    Raises ``ValueError`` when **another** rule holds the name, as
    :func:`create_rule` does. A rule's own name is not a collision:
    :func:`_name_taken` is asked to leave this row out, so re-typing it, or
    changing only its case, is an ordinary rename rather than a 409.

    ``modified`` is not written here. ``onupdate`` on the column advances it for
    any UPDATE the DAL issues, which is what makes it the last edit rather than
    the last one somebody remembered to record, and ``created`` is left alone --
    so the two differing is exactly "this rule has been renamed".

    ``modified_by`` has no such mechanism and so is written explicitly, from the
    identity the gateway forwarded. Optional because a request may carry none,
    and written on *every* rename including that one: leaving the previous
    editor in place would credit this edit to whoever made the last one.

    What the rule applies is untouched. The labels it wrote name it by id, so a
    renamed rule goes on holding every one of them, and a tag's page shows the
    new name against objects labelled under the old -- which is the point of
    naming the rule there rather than copying its name onto each row.
    """
    if _name_taken(name, exclude_id=rule_id):
        raise _taken(name)

    try:
        rows = store().query_write(
            s.rule.update()
            .where(s.rule.c.id == rule_id)
            .values(name=name, modified_by=modified_by)
            .returning(*_COLUMNS)
        )
    except IntegrityError as exc:
        # As on the create, and only the name rule: nothing else this statement
        # writes is constrained, so nothing else here can fail on one.
        if "uq_rule_name_lower" not in str(exc.orig):
            raise
        raise _taken(name) from exc

    if not rows:
        return None
    return {**rows[0], "tags": _tags_by_rule([rule_id])[rule_id]}


def _delete_tags_left_labelling_nothing(tag_ids: list[str]) -> None:
    """Delete whichever of *tag_ids* nothing holds any more.

    Called after a rule is deleted, over the tags that rule applied. A tag it
    was the only user of is left labelling nothing and named by no rule -- a
    word in the vocabulary that describes no part of the catalog -- so it goes
    with the rule that introduced it.

    The two ``NOT EXISTS`` are the whole of "nothing holds it", because they are
    the whole of what references a tag: ``tag_target`` for the objects carrying
    it and ``rule__tag`` for the rules applying it. The second is not an
    optimisation. Both cascade from ``tag``, so deleting a tag another rule
    still applies would silently strip that rule of it, and a rule with no tags
    applies nothing and could not have been created in the first place.

    Postgres decides which rows those are, rather than a read followed by a
    delete: the labels this rule wrote are gone by the time this runs, so the
    condition is about the rows that are left, and asking the database keeps the
    answer and the delete in one statement.
    """
    store().query_write(
        s.tag.delete().where(
            s.tag.c.id.in_(tag_ids),
            ~select(literal(1)).where(s.tag_target.c.tag_id == s.tag.c.id).exists(),
            ~select(literal(1)).where(s.rule__tag.c.tag_id == s.tag.c.id).exists(),
        )
    )


def delete_rule(rule_id: str, *, keep_tags: bool = False) -> bool:
    """Delete one rule, and say whether there was one to delete.

    ``False`` rather than an exception for an id that is not there: the settings
    page deletes from a list it has already read, so the interesting fact is
    that its list is stale, which the route turns into a 404.

    Most of what the delete does is a cascade declared in
    ``auto_ontology/dal/schema.py``. Its ``rule__tag`` rows go, and so do the **labels the
    rule applied**: ``tag_target.rule_id`` cascades, so deleting a rule takes
    back what it labelled and leaves hand-applied labels alone, which have no
    ``rule_id`` to cascade from. That is the point of deleting a rule rather
    than a property of how this is written -- a rule-applied tag is the rule
    still holding, and it goes on re-applying as the catalog grows until the
    rule is gone.

    **A tag the rule leaves labelling nothing goes too**, unless another rule
    applies it -- see :func:`_delete_tags_left_labelling_nothing`. Taking back
    the labels is what empties it, so cleaning up after that is finishing the
    same job: what is left otherwise is a word in the vocabulary that describes
    no part of the catalog and that nothing will apply again. A tag with a label
    anywhere else -- another object, another rule's doing, or a person's --
    stays, because it still says something.

    *keep_tags* is for a caller who wants the classification without the rule:
    the labels are handed to whoever wrote the rule -- ``rule.created_by``
    becomes their ``tagged_by`` and the ``rule_id`` is cleared, so the cascade
    finds nothing of its own to take. They read from then on as that person's
    own labels, which is the truest thing left to say about them: a rule is
    somebody deciding a search's results deserve a tag, and keeping the labels
    is standing by that decision after the mechanism is gone. No tag is emptied
    this way, so none is deleted either.

    Attributed rather than left to nobody because "nobody" reads as "Auto
    Generated", which would be the one wrong answer: a person chose these, and
    the record of who is right there in the rule as it is deleted. It is read
    from the row rather than taken from the caller -- a delete can be made by
    an admin who did not write the rule, and the labels were never theirs.

    Not that this reattaches them: an identical rule created afterwards does not
    pick them up, because :func:`auto_ontology.dal.tags.attach_tags_by_rule` leaves an
    existing row alone rather than claiming it -- the same rule that stops a
    rule from overwriting a hand-applied label. Which these now are.

    One transaction throughout. Half of this is worse than none of it: a failed
    delete would otherwise leave the labels detached from a rule still on
    screen, and a failed cleanup would leave the rule gone with tags nothing
    can explain.
    """
    with write_transaction():
        if keep_tags:
            store().query_write(
                s.tag_target.update()
                .where(s.tag_target.c.rule_id == rule_id)
                .values(
                    rule_id=None,
                    # A subquery rather than a read of its own: the rule is
                    # still there for another statement or two, and this way
                    # the author cannot be read from a row that changed
                    # underneath.
                    tagged_by=select(s.rule.c.created_by)
                    .where(s.rule.c.id == rule_id)
                    .scalar_subquery(),
                )
            )
            return bool(
                store().query_write(
                    s.rule.delete().where(s.rule.c.id == rule_id).returning(s.rule.c.id)
                )
            )

        # Read before the delete, which is the only moment the rule's tags can
        # be known: `rule__tag` cascades, so afterwards there is nothing left
        # saying which tags this rule was applying.
        applied = [
            row["tag_id"]
            for row in store().query_read(
                select(s.rule__tag.c.tag_id).where(s.rule__tag.c.rule_id == rule_id)
            )
        ]
        deleted = bool(
            store().query_write(
                s.rule.delete().where(s.rule.c.id == rule_id).returning(s.rule.c.id)
            )
        )
        if deleted and applied:
            _delete_tags_left_labelling_nothing(applied)
        return deleted
