# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tag CRUD, and the objects a tag labels.

One thing here is less obvious than it looks: **a tag name is unique
case-insensitively on the trimmed name**, and unlike ``zone.name`` that rule is
a real database constraint — ``uq_tag_name_lower`` in ``gsf/dal/schema.py``.

The duplicate checks in :func:`create_tag` and :func:`update_tag` therefore
exist for their error message rather than for correctness. The index is what
holds when two requests take the same name at once, and the check is what turns
the ordinary case into a readable 409 instead of a driver error. Both raise the
same ``ValueError``, so a caller has one behaviour to handle rather than two.

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
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import (
    Column,
    ColumnElement,
    FromClause,
    Select,
    Table,
    Text,
    cast,
    func,
    literal,
    null,
    select,
    union_all,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError

from gsf.dal import schema as s
from gsf.dal.session import store

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

#: ``created_by``/``modified_by`` for a tag no person asked for.
#:
#: Nothing writes it yet -- every tag today comes from the settings page, which
#: carries the caller's identity. It is reserved and rendered ("Auto Generated")
#: so that a future ingestion or rules path has an author to record that is not
#: a null, which already means "written before these columns existed".
#:
#: Safe as a literal: Better Auth generates its ids, so no account can hold it.
SYSTEM_ACTOR = "system"

#: ``type`` on a tagged item: which of the five things the row points at.
#:
#: Spelled again by ``TagTargetType`` in ``gsf/server/models.py``, which is the
#: enum an API caller sends, and by ``TagItemType`` in the frontend, which
#: labels the row from it. The API's copy is pinned to these by
#: ``gsf/server/tests/test_tag_target_type.py``.
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


def list_tags() -> list[dict[str, Any]]:
    """Every tag, ordered as the settings page renders them.

    Sorted case-insensitively with the id as a tie-break, so the order is total
    and a page does not reshuffle between two reads.
    """
    return store().query_read(
        select(*_COLUMNS).order_by(func.lower(s.tag.c.name), s.tag.c.id)
    )


def get_tag(tag_id: str) -> dict[str, Any] | None:
    """One tag by id, or ``None`` when no tag has that id.

    The same four columns the list returns, so the detail page and the row it
    was opened from show one tag rather than two projections of it.
    """
    rows = store().query_read(select(*_COLUMNS).where(s.tag.c.id == tag_id))
    return rows[0] if rows else None


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
    """
    return (
        select(
            item_id.label("id"),
            name.label("name"),
            path,
            literal(kind, Text).label("type"),
            s.tag_target.c.tagged.label("tagged"),
            *_parent_ids(parents or {}),
        )
        .select_from(joins)
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


def list_tag_targets(tag_id: str) -> list[dict[str, Any]]:
    """Every object carrying *tag_id*, as one list across the five kinds.

    Ordered by name case-insensitively with the id as a tie-break, matching
    :func:`list_tags`, so the order is total and does not depend on which
    branch of the union a row came from.

    An unknown *tag_id* yields ``[]`` rather than an error — the caller
    distinguishes a missing tag with :func:`get_tag`, which is the only read
    that can tell "no such tag" from "a tag with nothing tagged".
    """
    items = union_all(
        _term_targets(tag_id),
        _table_targets(tag_id),
        _column_targets(tag_id),
        _column_attribute_targets(tag_id),
        _sql_attribute_targets(tag_id),
    ).subquery()

    return store().query_read(
        select(items).order_by(func.lower(items.c.name), items.c.id)
    )


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
    header, or :data:`SYSTEM_ACTOR`. It is optional because the header is not
    guaranteed -- FastAPI is reachable directly on the private network -- and
    ``None`` records that nobody was named rather than failing the create over
    an audit field.

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


def attach_tag(*, tag_id: str, kind: str, item_id: str) -> list[dict[str, Any]] | None:
    """Label the *kind* object *item_id* with *tag_id*.

    Returns the object's tags afterwards, in the order a page renders them, so
    a caller redraws from this answer rather than reading membership back.

    Labelling something twice is not an error: the unique constraint the second
    insert hits is absorbed, and the answer is the same list either way. That
    matters because two clicks on the same tag are one user intent, and a 409
    would ask the page to explain a state it is already in.

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
            .values(tag_id=tag_id, **{column.name: item_id})
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
