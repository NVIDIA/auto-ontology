# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tag CRUD, and the objects a tag labels.

One thing here is less obvious than it looks: **a tag name is unique
case-insensitively on the trimmed name**, and unlike ``zone.name`` that rule is
a real database constraint — ``uq_tag_name_lower`` in ``auto_ontology/dal/schema.py``.

The duplicate checks in :func:`create_tag` and :func:`update_tag` therefore
exist for their error message rather than for correctness. The index is what
holds when two requests take the same name at once, and the check is what turns
the ordinary case into a readable 409 instead of a driver error. Both raise the
same ``ValueError``, so a caller has one behaviour to handle rather than two.

The tag list is paged, and :func:`list_tags` and :func:`count_tags` take the
same *search* so a page and its total describe one list. Paging is optional
there, unlike on a rule: the tag picker reads the whole vocabulary to filter it
in the browser, so an unpaged read is a first-class call rather than a fallback.

Membership is read in both directions, and the two reads are shaped for their
callers rather than for each other. :func:`list_tag_targets` unions the five
kinds of taggable object into *one* list of uniform rows — id, name, path, type,
tagged, and the ancestor ids below — so a tag's page renders a single table
rather than five, and can say "nothing is tagged" once. :func:`fetch_tags_map`
goes the other way, from a set of objects of *one* kind to the tags on each,
because that is what a detail page asks for.

The write side is :func:`attach_tag` and :func:`detach_tag`, and both return the
object's tags *after* the change rather than a bare success flag. That is what
lets a caller redraw from one answer instead of re-reading, and it is also how
each reports the case it cannot perform: ``None`` for an object or a tag that is
not there, which the router owes a 404 rather than a silent success.

A label also records **where it came from**: the account that applied it, or the
rule that matched. :func:`attach_tags_by_rule` is the second of those, writing a
whole rule's worth of labels in one statement per kind, and both reads carry the
source back so a tag's page can say who tagged each object. Whichever source
wrote a label first keeps it — the duplicate insert is absorbed rather than
overwriting — so a rule re-applying itself never claims a hand-applied label.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import (
    ARRAY,
    Column,
    ColumnElement,
    FromClause,
    Select,
    Subquery,
    Table,
    Text,
    and_,
    cast,
    exists,
    func,
    literal,
    not_,
    null,
    or_,
    select,
    true,
    union_all,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError

from auto_ontology.dal import schema as s
from auto_ontology.dal.session import store, write_transaction

#: The rule ``uq_tag_name_lower`` indexes, as a comparison the DAL can run.
_FOLDED_NAME = func.lower(func.trim(s.tag.c.name))

#: Every column the API returns for a tag, in one place so the list and the
#: create path cannot drift into returning different shapes.
_COLUMNS = (
    s.tag.c.id,
    s.tag.c.name,
    s.tag.c.created,
    s.tag.c.modified,
    s.tag.c.created_by,
    s.tag.c.modified_by,
)

#: ``type`` on a tagged item: which of the five things the row points at.
#:
#: Spelled again by ``TagTargetType`` in ``auto_ontology/server/models.py``, which is the
#: enum an API caller sends, and by ``TagItemType`` in the frontend, which
#: labels the row from it. The API's copy is pinned to these by
#: ``auto_ontology/server/tests/test_tag_target_type.py``.
TARGET_TERM = "term"
TARGET_TABLE = "table"
TARGET_COLUMN = "column"
TARGET_COLUMN_ATTRIBUTE = "column_attribute"
TARGET_SQL_ATTRIBUTE = "sql_attribute"

#: The one ``tag_target`` column each kind fills. Every function here that takes
#: a *kind* resolves it through this, so the mapping between the name an API
#: caller uses and the column ``exactly_one_target`` lets it set exists once.
_TARGET_COLUMNS: dict[str, Column] = {
    TARGET_TERM: s.tag_target.c.term_id,
    TARGET_TABLE: s.tag_target.c.table_id,
    TARGET_COLUMN: s.tag_target.c.column_id,
    TARGET_COLUMN_ATTRIBUTE: s.tag_target.c.column_attribute_id,
    TARGET_SQL_ATTRIBUTE: s.tag_target.c.sql_attribute_id,
}

#: Joins the ``path`` of a tagged item — where the object sits, as one string.
#: A dot rather than a prettier separator because these are qualified SQL
#: names, and that is the form a reader will type back into a query.
_PATH_SEPARATOR = "."

#: ``path`` for the kinds that have nowhere to point. Cast rather than a bare
#: ``NULL`` because this is one branch of a ``UNION``, and an untyped null there
#: leaves the column's type to be resolved from its neighbours.
_NO_PATH = cast(null(), Text).label("path")

#: The ancestors a tagged item is *addressed* through, as ids.
#:
#: ``path`` is the same relationship spelled for a reader; these are the same
#: relationship spelled for a URL, because the pages an item lives on are keyed
#: by id: a Table and a Column by their whole catalog chain, an attribute by the
#: Term whose page lists it. Deriving them on the client is not open to it — the
#: names in ``path`` are not unique, so there is nothing to look an id up by.
#:
#: Every branch of the union selects all four and fills only the ones its kind
#: has, so the columns line up and a caller has one shape to read.
_PARENT_ID_FIELDS = ("database_id", "schema_id", "table_id", "term_id")

#: The two attribute kinds and the link table each reaches its Term through.
#:
#: **Both are properties of at most one Term**, and that is the rule the rest of
#: this module leans on. ``uq_column_attribute__term_attribute_id`` and its
#: SqlAttribute twin are what hold it, which is why the reads here can join
#: rather than defend against a second Term that cannot exist.
#:
#: The consequence that matters here is that neither kind is taggable without a
#: Term, which :func:`attach_tag` enforces: every read that lists an attribute
#: joins through this link, so an unlinked one is invisible app-wide and its
#: page — a section of the Term's — has no address to open.
_ATTRIBUTE_TERM_LINKS: dict[str, tuple[Table, Table]] = {
    TARGET_COLUMN_ATTRIBUTE: (s.column_attribute, s.column_attribute__term),
    TARGET_SQL_ATTRIBUTE: (s.sql_attribute, s.sql_attribute__term),
}


def _target_column(kind: str) -> Column:
    """The ``tag_target`` column *kind* fills.

    Raises ``ValueError`` for a kind this table has no column for. That is a
    caller passing something the API never offered, not a missing object, which
    is why it is not the ``None`` the attach and detach paths return.
    """
    try:
        return _TARGET_COLUMNS[kind]
    except KeyError:
        known = ", ".join(sorted(_TARGET_COLUMNS))
        raise ValueError(
            f"Unknown tag target {kind!r}; expected one of {known}"
        ) from None


def _name_taken(name: str, *, exclude_id: str | None = None) -> bool:
    """Whether a tag already folds to *name* under ``uq_tag_name_lower``.

    *exclude_id* leaves one tag out, which is what a rename passes: a tag
    always collides with its own row, and the index would let the same write
    through for the same reason. Without it, re-typing a tag's own name -- or
    changing only its case -- would be reported as a name somebody else holds.
    """
    statement = select(s.tag.c.id).where(_FOLDED_NAME == name.strip().lower())
    if exclude_id is not None:
        statement = statement.where(s.tag.c.id != exclude_id)
    return bool(store().query_read(statement))


def _matching(search: str | None) -> list[ColumnElement[bool]]:
    """The WHERE for *search*, or nothing at all when there is none.

    A tag matches on its name, which is the whole of what a tag is -- the
    settings list shows nothing else to search by, so there is nothing else a
    query could match without hiding rows for a reason nothing on screen
    explains.

    Case-insensitive substring, as the rule list's ``q`` is. One function so
    :func:`list_tags` and :func:`count_tags` cannot come to disagree about what
    matches -- a page and a total taken from different filters would leave the
    list asking for rows that are not there.
    """
    if search is None or search.strip() == "":
        return []
    return [s.tag.c.name.ilike(f"%{search.strip()}%")]


def list_tags(
    *, search: str | None = None, skip: int = 0, limit: int | None = None
) -> list[dict[str, Any]]:
    """One page of tags, ordered as the settings page renders them.

    Sorted case-insensitively with the id as a tie-break, so the order is total
    and two pages of one list cannot repeat or skip a tag because the database
    reshuffled equal names between the requests.

    *skip* and *limit* select a window of that order, and :func:`count_tags` is
    the size of the whole match, which is what tells a caller when to stop
    asking. *limit* omitted returns every matching tag: the tag picker offers
    the whole vocabulary and filters it in the browser, so an unpaged read is
    its ordinary call rather than a special case.
    """
    statement = (
        select(*_COLUMNS)
        .where(*_matching(search))
        .order_by(func.lower(s.tag.c.name), s.tag.c.id)
        .offset(skip or None)
    )
    if limit is not None:
        statement = statement.limit(limit)

    return store().query_read(statement)


def count_tags(*, search: str | None = None) -> int:
    """The unpaged size of :func:`list_tags`, from the same filter."""
    rows = store().query_read(
        select(func.count(s.tag.c.id).label("total")).where(*_matching(search))
    )
    return int(rows[0]["total"]) if rows else 0


def get_tag(tag_id: str) -> dict[str, Any] | None:
    """One tag by id, or ``None`` when no tag has that id.

    The same four columns the list returns, so the detail page and the row it
    was opened from show one tag rather than two projections of it.
    """
    rows = store().query_read(select(*_COLUMNS).where(s.tag.c.id == tag_id))
    return rows[0] if rows else None


def existing_tag_ids(tag_ids: list[str]) -> set[str]:
    """Which of *tag_ids* name a tag, as a set.

    For a caller holding ids that have to be checked before it acts on them --
    saving a rule is the one -- so ids are all that comes back: a name or a
    timestamp would be read to be thrown away.

    Ids only, *and only the ones asked about*. Reading the vocabulary whole
    answers the same question, and did, but it ties the cost of checking three
    ids to how many tags a deployment has curated; this reads three rows by
    primary key whatever that number grows to.

    An empty list asks nothing and reads nothing: ``IN ()`` is not a query, and
    the answer is already in hand.
    """
    if not tag_ids:
        return set()
    rows = store().query_read(select(s.tag.c.id).where(s.tag.c.id.in_(tag_ids)))
    return {row["id"] for row in rows}


def _parent_ids(parents: dict[str, ColumnElement]) -> list[ColumnElement]:
    """The four ``_PARENT_ID_FIELDS`` columns, in order, nulled where absent.

    Cast for the reason :data:`_NO_PATH` is: an untyped ``NULL`` in one branch
    of a ``UNION`` takes its type from whichever branch happens to supply one.
    """
    return [
        parents.get(field, cast(null(), Text)).label(field)
        for field in _PARENT_ID_FIELDS
    ]


def _attribute_term(
    joined: FromClause, link_table: Table, attribute_id: ColumnElement
) -> tuple[FromClause, ColumnElement, ColumnElement]:
    """*joined* reaching through to the Term an attribute is a property of.

    Returns the extended join and the Term's ``(path, term_id)``. Used by
    **both** attribute kinds, so a ColumnAttribute and a SqlAttribute are the
    same row shape on a tag's page — the Term names where the attribute sits,
    and its id is what opens it. They are properties of a Term in exactly the
    same sense, and the page had no reason to tell them apart.

    A plain join, which ``uq_..._attribute_id`` on
    :data:`_ATTRIBUTE_TERM_LINKS` is what makes safe: at most one Term per
    attribute means at most one row out, and one tag on one object is one row.

    Outer, so an attribute with no Term is still listed — inert, with a null
    path and nowhere to open, which is what a row it cannot navigate should
    look like. :func:`attach_tag` refuses to create one, but a tag applied
    before that rule existed would otherwise vanish from its own tag's page
    with nothing to remove it by.

    A ColumnAttribute's own denormalised ``term_name`` would spare this join,
    but only by letting ``path`` name a Term that ``term_id`` does not — a row
    that says where it lives and then refuses to go there.
    """
    return (
        joined.outerjoin(
            link_table, link_table.c.attribute_id == attribute_id
        ).outerjoin(s.term, s.term.c.id == link_table.c.term_id),
        s.term.c.name.label("path"),
        s.term.c.id,
    )


def _targets(
    tag_id: str,
    item_id: ColumnElement,
    kind: str,
    name: ColumnElement,
    path: ColumnElement,
    joins: FromClause,
    parents: dict[str, ColumnElement] | None = None,
) -> Select:
    """One branch of :func:`list_tag_targets`, as the four share a shape.

    Each branch selects only its own rows without filtering for them: *joins*
    starts at the one ``tag_target`` column this kind fills, and a row of
    another kind has that column null — which is what ``exactly_one_target``
    guarantees, and why no ``WHERE`` on the kind is needed.

    *parents* names whichever of :data:`_PARENT_ID_FIELDS` this kind has;
    omitting it means none, which is a Term.

    ``tagged_by`` and ``rule_id``/``rule_name`` are the source of the label, for
    the "Tagged By" column: the account that applied it by hand, or the rule
    that matched. The rule is joined outer here rather than resolved by the
    caller, so one read answers the whole column -- a name per row fetched
    separately would be a query per tagged object.
    """
    return (
        select(
            item_id.label("id"),
            name.label("name"),
            path,
            literal(kind, Text).label("type"),
            s.tag_target.c.tagged.label("tagged"),
            s.tag_target.c.tagged_by.label("tagged_by"),
            s.tag_target.c.rule_id.label("rule_id"),
            s.rule.c.name.label("rule_name"),
            *_parent_ids(parents or {}),
        )
        .select_from(joins.outerjoin(s.rule, s.rule.c.id == s.tag_target.c.rule_id))
        .where(s.tag_target.c.tag_id == tag_id)
    )


def _term_targets(tag_id: str) -> Select:
    """Tagged Terms.

    No path and no ancestors: a Term is a glossary entry rather than a catalog
    object, so it sits under nothing there is a qualified name for, and its own
    page is addressed by its own id.
    """
    return _targets(
        tag_id,
        s.term.c.id,
        TARGET_TERM,
        s.term.c.name,
        _NO_PATH,
        s.tag_target.join(s.term, s.tag_target.c.term_id == s.term.c.id),
    )


def _table_targets(tag_id: str) -> Select:
    """Tagged Tables, under the ``database.schema`` that contains them."""
    return _targets(
        tag_id,
        s.catalog_table.c.id,
        TARGET_TABLE,
        s.catalog_table.c.name,
        (
            s.catalog_database.c.name
            + literal(_PATH_SEPARATOR, Text)
            + s.catalog_schema.c.name
        ).label("path"),
        s.tag_target.join(
            s.catalog_table, s.tag_target.c.table_id == s.catalog_table.c.id
        )
        .join(s.catalog_schema, s.catalog_table.c.schema_id == s.catalog_schema.c.id)
        .join(
            s.catalog_database,
            s.catalog_schema.c.database_id == s.catalog_database.c.id,
        ),
        {
            "database_id": s.catalog_database.c.id,
            "schema_id": s.catalog_schema.c.id,
        },
    )


def _column_targets(tag_id: str) -> Select:
    """Tagged Columns, under the ``database.schema.table`` that contains them."""
    separator = literal(_PATH_SEPARATOR, Text)
    return _targets(
        tag_id,
        s.catalog_column.c.id,
        TARGET_COLUMN,
        s.catalog_column.c.name,
        (
            s.catalog_database.c.name
            + separator
            + s.catalog_schema.c.name
            + separator
            + s.catalog_table.c.name
        ).label("path"),
        s.tag_target.join(
            s.catalog_column, s.tag_target.c.column_id == s.catalog_column.c.id
        )
        .join(s.catalog_table, s.catalog_column.c.table_id == s.catalog_table.c.id)
        .join(s.catalog_schema, s.catalog_table.c.schema_id == s.catalog_schema.c.id)
        .join(
            s.catalog_database,
            s.catalog_schema.c.database_id == s.catalog_database.c.id,
        ),
        {
            "database_id": s.catalog_database.c.id,
            "schema_id": s.catalog_schema.c.id,
            "table_id": s.catalog_table.c.id,
        },
    )


def _column_attribute_targets(tag_id: str) -> Select:
    """Tagged ColumnAttributes, under the Term they are a property of.

    Identical in shape to :func:`_sql_attribute_targets` — see
    :func:`_attribute_term`, which both call.
    """
    joins, path, term_id = _attribute_term(
        s.tag_target.join(
            s.column_attribute,
            s.tag_target.c.column_attribute_id == s.column_attribute.c.id,
        ),
        s.column_attribute__term,
        s.column_attribute.c.id,
    )
    return _targets(
        tag_id,
        s.column_attribute.c.id,
        TARGET_COLUMN_ATTRIBUTE,
        s.column_attribute.c.name,
        path,
        joins,
        {"term_id": term_id},
    )


def _sql_attribute_targets(tag_id: str) -> Select:
    """Tagged SqlAttributes.

    Qualified by its Term, exactly as a ColumnAttribute is: the two are
    properties of a Term in the same sense, so a tag's page had no reason to
    show them differently. A SqlAttribute reached through a CustomAnalysis
    rather than a Term cannot be tagged at all — see :func:`attach_tag` — so
    this never has to render one with nothing above it.
    """
    joins, path, term_id = _attribute_term(
        s.tag_target.join(
            s.sql_attribute, s.tag_target.c.sql_attribute_id == s.sql_attribute.c.id
        ),
        s.sql_attribute__term,
        s.sql_attribute.c.id,
    )
    return _targets(
        tag_id,
        s.sql_attribute.c.id,
        TARGET_SQL_ATTRIBUTE,
        s.sql_attribute.c.name,
        path,
        joins,
        {"term_id": term_id},
    )


def _all_targets(tag_id: str) -> Subquery:
    """The five kinds as one relation, which both the page and its total read.

    One function so :func:`list_tag_targets` and :func:`count_tag_targets`
    cannot come to disagree about which rows exist. They must agree exactly: a
    total larger than the rows a page can reach would have a caller asking for
    a next page forever, and the union is where a row can go missing -- each
    branch joins through to the object's name and its path, so an object whose
    chain is broken is absent from both halves rather than counted in one.
    """
    return union_all(
        _term_targets(tag_id),
        _table_targets(tag_id),
        _column_targets(tag_id),
        _column_attribute_targets(tag_id),
        _sql_attribute_targets(tag_id),
    ).subquery()


def list_tag_targets(
    tag_id: str, *, skip: int = 0, limit: int | None = None
) -> list[dict[str, Any]]:
    """One page of the objects carrying *tag_id*, across the five kinds.

    Ordered by name case-insensitively with the id as a tie-break, matching
    :func:`list_tags`, so the order is total and does not depend on which
    branch of the union a row came from -- which is what makes it pageable at
    all: two requests for two windows of an order the database was free to
    reshuffle would repeat rows and skip others.

    *skip* and *limit* select a window of that order, and *limit* omitted
    returns every object. :func:`count_tag_targets` is the size of the whole
    list, which is what tells a caller when to stop asking.

    Each row carries the source of its label: ``tagged_by``, the account that
    applied it by hand, and ``rule``, the rule that applied it instead. Exactly
    one of the two is set on a row written since the columns existed; both are
    null on an older one, which reads as a label of unknown origin.

    An unknown *tag_id* yields ``[]`` rather than an error — the caller
    distinguishes a missing tag with :func:`get_tag`, which is the only read
    that can tell "no such tag" from "a tag with nothing tagged".
    """
    items = _all_targets(tag_id)
    statement = (
        select(items)
        .order_by(func.lower(items.c.name), items.c.id)
        .offset(skip or None)
    )
    if limit is not None:
        statement = statement.limit(limit)

    return [_with_rule(row) for row in store().query_read(statement)]


def count_tag_targets(tag_id: str) -> int:
    """How many objects carry *tag_id*, which is the unpaged size of the list.

    Counted over the same union the page is taken from rather than over
    ``tag_target`` directly: a label whose object cannot be joined through is
    not a row the list can show, and counting it would leave a caller a page
    short of a total it can never reach.
    """
    rows = store().query_read(
        select(func.count().label("total")).select_from(_all_targets(tag_id))
    )
    return int(rows[0]["total"]) if rows else 0


def _with_rule(row: dict[str, Any]) -> dict[str, Any]:
    """*row* with its two rule columns folded into one nested rule, or null.

    Nested rather than left as ``rule_id``/``rule_name`` because "no rule
    applied this" is one fact, and two columns to check for it is two ways for
    a reader to get it half right.
    """
    rule_id = row.pop("rule_id")
    rule_name = row.pop("rule_name")
    row["rule"] = None if rule_id is None else {"id": rule_id, "name": rule_name}
    return row


def fetch_tags_map(kind: str, item_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    """``{item_id: [tag, ...]}`` for objects of one *kind*.

    One query for the whole set, so a page listing a term's attributes reads
    their tags once rather than once per row.

    Only id and name: this is the chip a detail page draws beside the object,
    and the timestamps :func:`list_tags` returns describe the tag rather than
    the labelling. Objects with no tags are absent from the map rather than
    present with an empty list, which is what makes ``.get(id, [])`` at the call
    site the same expression whether the object was queried for or not.
    """
    column = _target_column(kind)
    if not item_ids:
        return {}

    rows = store().query_read(
        select(column.label("item_id"), s.tag.c.id, s.tag.c.name)
        .select_from(s.tag_target.join(s.tag, s.tag.c.id == s.tag_target.c.tag_id))
        .where(column.in_(item_ids))
        .order_by(func.lower(s.tag.c.name), s.tag.c.id)
    )

    tags: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        tags.setdefault(row["item_id"], []).append(
            {"id": row["id"], "name": row["name"]}
        )
    return tags


def create_tag(*, name: str, created_by: str | None = None) -> dict[str, Any]:
    """Create a tag and return it.

    *name* is stored as given; the caller is expected to have trimmed it. Raises
    ``ValueError`` when the name is already taken, whether that is caught by the
    check or by the unique index underneath it.

    *created_by* is the Better Auth user id the router read from the gateway's
    header. It is optional because the header is not guaranteed -- FastAPI is
    reachable directly on the private network -- and ``None`` records that
    nobody was named rather than failing the create over an audit field. The
    pages read that as "Auto Generated": with no account to name, the
    deployment is what is left.

    ``modified_by`` is deliberately *not* set alongside it. It answers "who last
    edited this", and a tag that has only ever been created has not been edited
    -- the same reason ``modified`` equalling ``created`` reads as "Never".
    """
    if _name_taken(name):
        raise ValueError(f"Tag with name {name!r} already exists")

    try:
        rows = store().query_write(
            s.tag.insert().values(name=name, created_by=created_by).returning(*_COLUMNS)
        )
    except IntegrityError as exc:
        # Only the name rule is translated. A different constraint failing means
        # something this function does not model, and hiding it behind "already
        # exists" would send the caller after the wrong bug.
        if "uq_tag_name_lower" not in str(exc.orig):
            raise
        raise ValueError(f"Tag with name {name!r} already exists") from exc

    return rows[0]


def update_tag(
    *, tag_id: str, name: str, modified_by: str | None = None
) -> dict[str, Any] | None:
    """Rename a tag and return it. ``None`` when no tag has that id.

    The name is a tag's only editable part, so this takes it as a plain
    argument rather than a dict of updates: there is no second field for a
    caller to leave out, and no distinction between "not provided" and "set to
    nothing" to model. *name* is stored as given, as on :func:`create_tag` --
    the caller is expected to have trimmed it.

    Raises ``ValueError`` when **another** tag holds the name, from the check or
    from ``uq_tag_name_lower`` underneath it. A tag's own name is not a
    collision: :func:`_name_taken` is asked to leave this row out, so re-typing
    it, or changing only its case, is an ordinary rename rather than a 409.

    ``modified`` is not written here. ``onupdate`` on the column advances it for
    any UPDATE the DAL issues, which is what makes it the *last edit* rather
    than the last edit somebody remembered to record, and ``created`` is left
    alone -- so the two differing is exactly "this tag has been renamed", which
    is what the settings page renders.

    ``modified_by`` has no such mechanism and so is written explicitly, from the
    identity the gateway forwarded. It is optional for the reason ``created_by``
    is on :func:`create_tag`, and it is set on *every* rename including one that
    names nobody: leaving the previous editor in place would credit this edit to
    whoever made the last one.

    The rows it labels are untouched: membership is keyed by id, so a rename
    reaches every object carrying the tag without a single ``tag_target`` row
    being written.
    """
    if _name_taken(name, exclude_id=tag_id):
        raise ValueError(f"Tag with name {name!r} already exists")

    try:
        rows = store().query_write(
            s.tag.update()
            .where(s.tag.c.id == tag_id)
            .values(name=name, modified_by=modified_by)
            .returning(*_COLUMNS)
        )
    except IntegrityError as exc:
        # Translated for the reason `create_tag` translates it, and only the
        # name rule: nothing else this statement writes is constrained, so
        # nothing else here can fail on a rule this function models.
        if "uq_tag_name_lower" not in str(exc.orig):
            raise
        raise ValueError(f"Tag with name {name!r} already exists") from exc

    return rows[0] if rows else None


def delete_tag(tag_id: str) -> bool:
    """Delete a tag. ``False`` when no tag has that id.

    The boolean is what separates "deleted" from "was never there" for the
    router, which owes the caller a 404 rather than a silent 204 for the second.

    Its ``tag_target`` rows go with it by cascade, so deleting a tag unlabels
    everything it labelled rather than leaving rows pointing at nothing.
    """
    rows = store().query_write(
        s.tag.delete().where(s.tag.c.id == tag_id).returning(s.tag.c.id)
    )
    return bool(rows)


def _attribute_without_a_term(kind: str, item_id: str) -> bool:
    """Whether *item_id* is an attribute of that *kind* that no Term claims.

    ``False`` for an attribute that is **not there at all**, which is a
    different answer the caller owes a 404 for: the insert in
    :func:`attach_tag` reports that one through its foreign key, and reporting
    it from here instead would call a missing object ineligible.

    One extra read, and only for the two attribute kinds. The alternative is a
    label on an object nothing can display, which no read would ever return and
    no page could ever remove.
    """
    tables = _ATTRIBUTE_TERM_LINKS.get(kind)
    if tables is None:
        return False
    attribute, link = tables
    rows = store().query_read(
        select(
            select(literal(1))
            .where(link.c.attribute_id == attribute.c.id)
            .exists()
            .label("has_term")
        ).where(attribute.c.id == item_id)
    )
    return bool(rows) and not rows[0]["has_term"]


def attach_tag(
    *, tag_id: str, kind: str, item_id: str, tagged_by: str | None = None
) -> list[dict[str, Any]] | None:
    """Label the *kind* object *item_id* with *tag_id*.

    Returns the object's tags afterwards, in the order a page renders them, so
    a caller redraws from this answer rather than reading membership back.

    *tagged_by* is the account doing it, recorded for the "Tagged By" column on
    the tag's page. Optional, and omitting it stores a null, which with a null
    ``rule_id`` beside it is what that column reads as "Auto Generated": a
    label no account and no rule is claiming came from the deployment itself.

    Labelling something twice is not an error: the unique constraint the second
    insert hits is absorbed, and the answer is the same list either way. That
    matters because two clicks on the same tag are one user intent, and a 409
    would ask the page to explain a state it is already in.

    The absorbed write keeps the row already there, so the source of a label is
    whoever applied it *first*: a person re-applying what a rule matched does
    not take it over, and a rule matching what a person labelled does not take
    it from them.

    ``None`` when the tag or the object does not exist. Both are foreign keys,
    so one round trip establishes it -- no existence check is needed on the path
    that succeeds, which is the ordinary one.

    Raises ``ValueError`` for an **attribute of no Term**, of either kind. The
    object exists, so this is not the ``None`` above; it is ineligible, for the
    reason on :data:`_ATTRIBUTE_TERM_LINKS`. Reachable only by a caller going
    straight to the API — an attribute with no Term appears in no list the UI
    can offer a tag picker from.
    """
    column = _target_column(kind)
    if _attribute_without_a_term(kind, item_id):
        raise ValueError(
            f"{kind} {item_id!r} is not a property of any term, so it cannot be tagged"
        )
    try:
        store().query_write(
            insert(s.tag_target)
            .values(
                tag_id=tag_id,
                tagged_by=tagged_by,
                **{column.name: item_id},
            )
            .on_conflict_do_nothing()
        )
    except IntegrityError as exc:
        # A foreign key is the only constraint left to fail here: the unique
        # ones are handled above, and `exactly_one_target` cannot trip when
        # exactly one column is set. Anything else is a bug worth surfacing.
        if "fk_tag_target" not in str(exc.orig):
            raise
        return None

    return fetch_tags_map(kind, [item_id]).get(item_id, [])


def attach_tags_by_rule(
    *, rule_id: str, tag_ids: list[str], targets: list[tuple[str, str]]
) -> int:
    """Label every object in *targets* with every tag in *tag_ids*, as a rule.

    *targets* is ``(kind, item_id)`` pairs, which is what replaying a rule's
    search yields once the kinds it cannot label are dropped. Returns how many
    labels this actually applied, which is the point of the ``RETURNING``: the
    number a caller can log or report without reading membership back.

    One statement per kind rather than one per label. A rule matching a few
    hundred objects across a handful of tags is thousands of rows, and a round
    trip each would make creating a rule a request that visibly hangs.

    Already-labelled objects are skipped, not overwritten -- see
    :func:`attach_tag` on why the first source of a label keeps it -- so
    re-applying a rule as the catalog grows costs nothing for what it labelled
    last time, and never takes a hand-applied label away from the person who
    applied it.

    ``tagged_by`` is left null: a rule is not an account, and ``rule_id`` is
    what says a rule did this. The two are read together, so a row is never
    ambiguous about which it was.

    Attributes of no Term are not defended against, unlike :func:`attach_tag`:
    every search that could produce one joins through the Term link, so a rule
    replaying a search cannot see one to pass here.
    """
    if not tag_ids or not targets:
        return 0

    by_kind: dict[str, list[str]] = {}
    for kind, item_id in targets:
        by_kind.setdefault(kind, []).append(item_id)

    applied = 0
    for kind, item_ids in by_kind.items():
        column = _target_column(kind)
        rows = [
            {"tag_id": tag_id, "rule_id": rule_id, column.name: item_id}
            for tag_id in dict.fromkeys(tag_ids)
            for item_id in dict.fromkeys(item_ids)
        ]
        applied += len(
            store().query_write(
                insert(s.tag_target)
                .values(rows)
                .on_conflict_do_nothing()
                .returning(s.tag_target.c.id)
            )
        )
    return applied


def apply_labels_now_matched(
    *, rule_id: str, tag_ids: list[str], targets: dict[str, Select]
) -> int:
    """``INSERT ... SELECT`` the rule's tags onto everything it matches.

    The second half of a replay; :func:`remove_labels_no_longer_matched` is the
    first, and ``auto_ontology.server.rules.service.reapply_rules`` is what runs
    the two -- every rule's removal before any rule's application, for a reason
    that is on *that* function rather than here, because neither half can see
    it on its own.

    *targets* is a ``SELECT id`` per kind, as
    :func:`auto_ontology.dal.search.matching_id_selects` produces it — statements, not
    ids. They are embedded in the write rather than run first, so a rule
    matching fifty thousand columns never moves an id out of Postgres: the
    match, the comparison against what is already labelled, and the insert are
    one plan per statement. That is what makes this affordable on every ingest
    rather than only on the small rules.

    Returns labels written, not objects matched -- an object already carrying
    the tag is untouched and uncounted, which is what makes a run over an
    unchanged catalog report zero rather than reporting the whole catalog
    again. It is also what keeps ``tag_target.tagged`` honest: a row left alone
    keeps the day it was really applied, where a wipe and a rewrite would move
    every label's "Tagged" to the date of the last ingest.

    Tags no longer in *tag_ids* are not considered, because they cannot occur:
    a rule's tags are fixed at creation (see :func:`auto_ontology.dal.rules.update_rule`)
    and a tag deleted from the vocabulary takes its labels with it by cascade.

    One transaction over the kinds, which is as much as is worth holding: five
    statements that are one rule's labels, and a rule applied to Columns but
    not to the Terms it also matched is a state no reader should see.

    One statement per kind, which is as far as this can be collapsed:
    ``tag_target`` has a column per kind, so two kinds cannot share an INSERT.

    The tags are *not* a statement each. They arrive as an array and are
    cross-joined to the match, so the work is one pass whatever a rule carries.
    Both of the things that make that cheap matter, and dropping either brings
    back a cost proportional to the tags -- measured over half a million
    columns with a hundred thousand of them matching, on a pass where every
    label was already in place:

    The ``NOT EXISTS`` is the larger one, 1.6s down to 0.4s for three tags.
    Without it the conflict clause is still correct but arrives too late: every
    matched row is built and offered to the index before being thrown away, and
    on a settled catalog that is the whole hundred thousand, every night, for
    nothing. Leaving with the row not yet formed skips all of it.

    ``unnest`` is what keeps that flat: one tag, three and five all measured
    0.4s, where a statement each would have paid the 0.4s five times. Rules
    multiply the same way, so on a deployment with tens of rules the two
    together are the difference between a nightly pass of seconds and one of
    minutes -- five tags were 2.7s before this and 0.4s after.

    Neither helps the *first* pass, which really does write a row per pair --
    three tags over that catalog took eight seconds either way, the ``NOT
    EXISTS`` costing about 6% for an anti-join that finds nothing. That is the
    right trade: the first pass happens once and the idle one every night
    after.

    Not a materialized CTE. Reading the match once per tag sounds like the
    thing to avoid, but the scan is 73ms of the above and pinning it measured
    no better -- the cost is in the rows, not in finding them.

    Repeats collapse before the array is built: two clicks on one tag are one
    intention, as :func:`attach_tags_by_rule` treats them, and the duplicate
    would otherwise survive the anti-join, which only knows what is already
    stored.

    ``on_conflict_do_nothing`` stays behind the anti-join as the guarantee
    rather than the filter: the two read the same ``(tag_id, <kind>)`` unique
    constraints, but a label applied by hand between this statement's snapshot
    and its write would slip past the first and has to not raise. It is also
    what keeps such a label with the person who applied it -- see
    :func:`attach_tags_by_rule`, which absorbs the duplicate for the same
    reason. ``id`` is left to the column default.
    """
    if not tag_ids:
        return 0

    # ``render_derived`` and not ``alias``, as in ``search._synonym_exists``:
    # the alias alone names the function's result without naming its column,
    # and ``rule_tag.tag_id`` is then an undefined column at runtime.
    tags = (
        func.unnest(cast(list(dict.fromkeys(tag_ids)), ARRAY(Text)))
        .table_valued("tag_id")
        .render_derived("rule_tag")
    )
    stored = s.tag_target.alias("stored")

    applied = 0
    with write_transaction():
        for kind, matched in targets.items():
            column = _target_column(kind)
            found = matched.subquery()
            applied += len(
                store().query_write(
                    insert(s.tag_target)
                    .from_select(
                        ["tag_id", "rule_id", column.name],
                        # ``ON true``: the cross product of the tags and the
                        # match *is* what "apply these tags to these objects"
                        # means. Spelled as a join rather than as two FROM
                        # entries only so it reads as deliberate -- SQLAlchemy
                        # warns about the latter, on the assumption that a
                        # cartesian product is a forgotten condition.
                        select(tags.c.tag_id, literal(rule_id), found.c.id)
                        .select_from(found.join(tags, true()))
                        .where(
                            # Deliberately blind to `rule_id`: what the
                            # constraint forbids is the tag twice on the
                            # object, whoever put it there, so this has to ask
                            # the same question.
                            not_(
                                exists(
                                    select(literal(1)).where(
                                        stored.c.tag_id == tags.c.tag_id,
                                        stored.c[column.name] == found.c.id,
                                    )
                                )
                            )
                        ),
                    )
                    .on_conflict_do_nothing()
                    .returning(s.tag_target.c.id)
                )
            )
    return applied


def remove_labels_no_longer_matched(*, rule_id: str, targets: dict[str, Select]) -> int:
    """Delete the rule's labels on objects its search no longer finds.

    The first half of a replay, and it runs for *every* rule before
    :func:`apply_labels_now_matched` runs for any -- see
    ``auto_ontology.server.rules.service.reapply_rules``, which is what orders
    the two and where the reason lives.

    An empty *targets* removes every label the rule has. A rule whose search
    matches nothing is a rule that should be labelling nothing, which is an
    ordinary state -- the catalog may not have grown into it yet, or may have
    grown out of it -- and not a reason to leave yesterday's labels standing.

    **What is removed is this rule's doing and nothing else.** The statement is
    scoped to ``rule_id``, and a hand-applied label carries none -- so an object
    a person tagged by hand is not reachable from here, even when the rule also
    matched it and even when it stops matching. The same holds the other way: a
    label this rule applied to an object that has since been renamed out of the
    search is removed, because a rule-applied tag is the rule still holding
    rather than a fact of its own.

    One statement for every kind at once, unlike the insert: what is deleted is
    identified by ``rule_id`` and the columns are only a test, so all five fit
    in one WHERE.

    That test is "kept", negated. A row is kept when the column for its kind is
    the one it fills *and* its id is still in that kind's match; anything else
    the rule owns goes, which covers the three ways a label goes stale -- the
    object stopped matching, its whole kind dropped out of the rule's filters,
    or the rule now matches nothing at all.

    Written as ``IS NOT NULL AND IN`` rather than ``IN`` alone because a row
    fills exactly one of the five columns and leaves the rest null: ``NULL IN
    (...)`` is null, and a null inside the negation would leave the row
    undecided rather than deleted.
    """
    kept = [
        and_(_target_column(kind).is_not(None), _target_column(kind).in_(matched))
        for kind, matched in targets.items()
    ]
    conditions: list[ColumnElement[bool]] = [s.tag_target.c.rule_id == rule_id]
    if kept:
        conditions.append(not_(or_(*kept)))
    return len(
        store().query_write(
            s.tag_target.delete().where(*conditions).returning(s.tag_target.c.id)
        )
    )


def detach_tag(*, tag_id: str, kind: str, item_id: str) -> list[dict[str, Any]] | None:
    """Take *tag_id* off the *kind* object *item_id*.

    Returns the object's remaining tags, as :func:`attach_tag` does, so removing
    a chip and adding one are the same shape for the caller.

    ``None`` when there was no such label to remove -- an unknown tag, an
    unknown object, and a pair that was never linked are one case here, and all
    three mean the page acted on a chip that is no longer there.
    """
    column = _target_column(kind)
    rows = store().query_write(
        s.tag_target.delete()
        .where(s.tag_target.c.tag_id == tag_id, column == item_id)
        .returning(s.tag_target.c.id)
    )
    if not rows:
        return None

    return fetch_tags_map(kind, [item_id]).get(item_id, [])
