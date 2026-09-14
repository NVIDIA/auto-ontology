# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""API route handlers for tags."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from gsf.dal import tags as dal
from gsf.server.identity import resolve_internal_user
from gsf.server.models import TagTargetType
from gsf.server.pagination import LIMIT_QUERY, SKIP_QUERY
from gsf.server.responses import (
    IdResponse,
    TagChipListResponse,
    TagItemPageResponse,
    TagPageResponse,
    TagResponse,
)

router = APIRouter()

#: Longest name a tag may have, counted after trimming. Mirrored by the create
#: dialog, which shows it as "(Max. 25)" and stops typing there — this is the
#: half that a caller bypassing the UI still meets.
MAX_TAG_NAME_LENGTH = 25


class TagCreate(BaseModel):
    name: str


class TagUpdate(BaseModel):
    """A rename. The name is the only part of a tag there is to edit."""

    name: str


class TagTarget(BaseModel):
    """The object to label: what kind it is, and which one."""

    type: TagTargetType
    id: str


@router.get("/tags", response_model=TagPageResponse)
def list_tags(
    q: str | None = Query(
        default=None,
        description="Case-insensitive substring filter on the tag name.",
    ),
    skip: int = SKIP_QUERY,
    limit: int | None = LIMIT_QUERY,
) -> dict:
    """Return tags, ordered by name.

    *q*, when given, keeps the tags whose name contains it -- a name is the
    whole of what the settings list shows, so there is nothing else to search
    by.

    *skip*/*limit* select one page of that order, and ``total`` counts every
    match so a caller knows when to stop asking. Omitting *limit* returns every
    matching tag, which is what the tag picker asks for: it offers the whole
    vocabulary on every detail page and narrows it in the browser.
    """
    rows = dal.list_tags(search=q, skip=skip, limit=limit)
    # The count is a second read, so it is worth skipping for the request that
    # asked for everything: a whole unpaged list already is its own total. Same
    # bargain the rule list and the targets list below make.
    total = dal.count_tags(search=q) if skip or limit is not None else len(rows)
    return {"data": rows, "count": len(rows), "total": total}


@router.get("/tags/{tag_id}", response_model=TagResponse)
def get_tag(tag_id: str) -> dict:
    """Return one tag.

    The tag alone: what it labels is a list of its own, read a page at a time
    from :func:`list_tag_targets` below. A tag can label the whole catalog --
    that is what a rule does to one -- so a detail that carried every object
    would grow without bound while the page it feeds shows a screenful.

    404 for an unknown id, rather than an empty tag: the page opens this from a
    row it has already read, so a missing tag means the row is stale, and a
    blank page titled with nothing would look like a tag with no items.
    """
    tag = dal.get_tag(tag_id)
    if tag is None:
        raise HTTPException(status_code=404, detail=f"Tag {tag_id!r} not found")
    return {"data": tag}


@router.get("/tags/{tag_id}/targets", response_model=TagItemPageResponse)
def list_tag_targets(
    tag_id: str,
    skip: int = SKIP_QUERY,
    limit: int | None = LIMIT_QUERY,
) -> dict:
    """Return the objects this tag labels, as one page across the five kinds.

    Ordered by name, case-insensitively, with *skip*/*limit* selecting one
    window of that order; ``total`` counts everything the tag labels, so a
    caller knows when to stop asking. Omitting *limit* returns every object,
    which is what a caller wanting the whole list in one answer asks for.

    Each row says where its label came from -- the account that applied it or
    the rule that matched -- which is the "Tagged By" column on the tag's page.

    404 for an unknown tag rather than an empty page: the page reads this beside
    :func:`get_tag`, and a tag that is gone should say so once rather than draw
    as a tag with nothing tagged. An *existing* tag with nothing tagged is the
    opposite case and answers 200 with an empty ``data``, which the page renders
    as its empty state.
    """
    if dal.get_tag(tag_id) is None:
        raise HTTPException(status_code=404, detail=f"Tag {tag_id!r} not found")

    rows = dal.list_tag_targets(tag_id, skip=skip, limit=limit)
    # The count is a second read over the same union, so it is worth skipping
    # for the request that asked for everything: an unpaged list is its own
    # total. Same bargain the rule list makes.
    total = dal.count_tag_targets(tag_id) if skip or limit is not None else len(rows)
    return {"data": rows, "count": len(rows), "total": total}


def _validated_name(raw: str) -> str:
    """*raw* trimmed, or a 400 saying what is wrong with it.

    Shared by create and rename so one name is acceptable on both, rather than
    a tag being creatable under a name a rename would reject. Trimming happens
    before anything else, so the name that is length-checked,
    uniqueness-checked and stored is the one a reader sees.
    """
    name = raw.strip()
    if name == "":
        raise HTTPException(status_code=400, detail="Tag name is required")
    if len(name) > MAX_TAG_NAME_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Tag name must be at most {MAX_TAG_NAME_LENGTH} characters",
        )
    return name


@router.post("/tags", status_code=201, response_model=TagResponse)
def create_tag(request: Request, body: TagCreate) -> dict:
    """Create a tag.

    The author is read from the gateway's identity header rather than taken
    from the body: the Next.js route in front of this one has already resolved
    the session, so the header is the one account that can be credited, and a
    body field would let a caller name somebody else.

    ``required=False`` because the identity is audit rather than authorization
    -- the route's permission gate is the gateway's -- so a direct call on the
    private network still creates the tag, with no author recorded.
    """
    try:
        row = dal.create_tag(
            name=_validated_name(body.name),
            created_by=resolve_internal_user(request, required=False),
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"data": row}


@router.patch("/tags/{tag_id}", response_model=TagResponse)
def update_tag(request: Request, tag_id: str, body: TagUpdate) -> dict:
    """Rename a tag.

    Answers with the whole tag rather than an echo of the name, because the
    rename has also advanced ``modified`` and recorded a ``modified_by`` — so
    the settings page redraws the row it just edited from this one response
    instead of re-reading the list to find out what they became.

    The editor is read from the gateway's identity header, for the reason
    ``create_tag`` reads the author from it.

    404 for an unknown id and 409 for a name another tag holds, as create and
    delete answer: the page renames from a list it has already read, so either
    means that list is stale.

    A rename to the tag's own name is not a 409 — see ``update_tag`` in the DAL
    — and neither is one that only changes case, which is the ordinary way to
    fix a tag that was created shouting.
    """
    try:
        row = dal.update_tag(
            tag_id=tag_id,
            name=_validated_name(body.name),
            modified_by=resolve_internal_user(request, required=False),
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if row is None:
        raise HTTPException(status_code=404, detail=f"Tag {tag_id!r} not found")
    return {"data": row}


@router.delete("/tags/{tag_id}", response_model=IdResponse)
def delete_tag(tag_id: str) -> dict:
    """Delete one tag by id.

    404 rather than 204 for an id that is not there: the settings page deletes
    from a list it has already read, so a missing tag means its list is stale,
    and answering "done" would leave the row on screen with nothing to explain
    it.
    """
    if not dal.delete_tag(tag_id):
        raise HTTPException(status_code=404, detail=f"Tag {tag_id!r} not found")
    return {"data": {"id": tag_id}}


def _chips(tags: list[dict]) -> dict:
    return {"data": tags, "count": len(tags)}


def _target_or_tag_missing(
    tag_id: str, target_type: TagTargetType, item_id: str
) -> HTTPException:
    """Name whichever side is gone, for the attach path's 404.

    The DAL reports both as one ``None``, since a single insert establishes it
    without reading either row first. Which one it was costs a query only on
    this path, which is the one that already failed.
    """
    if dal.get_tag(tag_id) is None:
        detail = f"Tag {tag_id!r} not found"
    else:
        detail = f"{target_type.value} {item_id!r} not found"
    return HTTPException(status_code=404, detail=detail)


@router.post(
    "/tags/{tag_id}/targets", status_code=201, response_model=TagChipListResponse
)
def attach_tag(request: Request, tag_id: str, body: TagTarget) -> dict:
    """Label one object with this tag.

    Both write routes live here rather than on ``/terms/{id}/tags`` and its two
    attribute equivalents, because ``tag_target`` is one polymorphic table and
    six routes over three routers would spread that polymorphism across the
    codebase to say the same thing.

    Answers with the **object's** tags, so the page that added a chip redraws
    its whole set from one response instead of trusting its own optimistic
    edit. Labelling something that already carries the tag is a 201 with the
    same body: two clicks on one tag are one intention.

    Who applied the label is read from the gateway's identity header, for the
    reason ``create_tag`` reads an author from it, and shows in the "Tagged By"
    column on the tag's page. ``required=False`` for the same reason too: it is
    audit rather than authorization, so a direct call on the private network
    still labels the object -- recorded as the deployment's own doing, which
    that column names "Auto Generated". See ``attach_tag`` in the DAL.

    Re-labelling something keeps the source it already had, so this does not
    take a label away from the rule that applied it -- see ``attach_tag`` in the
    DAL.

    409 for a column or SQL attribute that is not a property of any term: it
    exists, so a 404 would be a lie, but it is unreachable — every list of
    attributes goes through the term that owns them — and a tag on it could
    never be seen or taken off again.
    """
    try:
        tags = dal.attach_tag(
            tag_id=tag_id,
            kind=body.type,
            item_id=body.id,
            tagged_by=resolve_internal_user(request, required=False),
        )
    except ValueError as exc:
        # Only the DAL's eligibility rule reaches here. Its other ``ValueError``
        # is for an unknown target kind, which ``TagTargetType`` has already
        # rejected with a 422 before this function runs.
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if tags is None:
        raise _target_or_tag_missing(tag_id, body.type, body.id)
    return _chips(tags)


@router.delete(
    "/tags/{tag_id}/targets/{target_type}/{item_id}",
    response_model=TagChipListResponse,
)
def detach_tag(tag_id: str, target_type: TagTargetType, item_id: str) -> dict:
    """Take this tag off one object.

    404 when the object was not carrying it — including when either side no
    longer exists. The page removed a chip it had just rendered, so all three
    mean its view is stale, and answering "done" would leave the chip gone from
    the screen and still on the object.
    """
    tags = dal.detach_tag(tag_id=tag_id, kind=target_type, item_id=item_id)
    if tags is None:
        raise HTTPException(
            status_code=404,
            detail=f"Tag {tag_id!r} is not on {target_type.value} {item_id!r}",
        )
    return _chips(tags)
