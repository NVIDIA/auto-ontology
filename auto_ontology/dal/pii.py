# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistence operations for automatic PII classification."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

from sqlalchemy import or_
from sqlalchemy import func, select

from auto_ontology.dal import schema as s
from auto_ontology.dal.session import store

PII_TAG_NAME = "PII"
_MAX_SAMPLE_RETENTION_DAYS = 365


def sample_retention_days() -> int:
    """Configured retention period, bounded to one year."""

    try:
        configured = int(os.environ.get("SAMPLE_VALUE_RETENTION_DAYS", "30"))
    except (TypeError, ValueError):
        configured = 30
    return min(max(configured, 1), _MAX_SAMPLE_RETENTION_DAYS)


def mark_columns_pii_processed(column_ids: list[str]) -> int:
    """Mark successfully classified catalog columns as processed."""

    if not column_ids:
        return 0

    rows = store().query_write(
        s.catalog_column.update()
        .where(
            s.catalog_column.c.id.in_(column_ids),
            s.catalog_column.c.pii_processed.is_(False),
        )
        .values(pii_processed=True)
        .returning(s.catalog_column.c.id)
    )
    return len(rows)


def delete_column_sample_embeddings(column_ids: list[str]) -> int:
    """Delete data and semantic VDB rows that may contain column samples."""

    if not column_ids:
        return 0

    from auto_ontology.vdb import get_data_vdb, get_semantic_vdb

    attribute_rows = store().query_read(
        select(s.column__has_attribute.c.attribute_id).where(
            s.column__has_attribute.c.column_id.in_(column_ids)
        )
    )
    data_vdb = get_data_vdb()
    semantic_vdb = get_semantic_vdb()
    deleted = sum(data_vdb.delete_by_id(column_id) for column_id in set(column_ids))
    deleted += sum(
        semantic_vdb.delete_by_id(attribute_id)
        for attribute_id in {row["attribute_id"] for row in attribute_rows}
    )
    return deleted


def clear_column_sample_values(column_ids: list[str]) -> int:
    """Delete persisted samples for columns that must not expose values."""

    if not column_ids:
        return 0
    delete_column_sample_embeddings(column_ids)
    rows = store().query_write(
        s.catalog_column.update()
        .where(
            s.catalog_column.c.id.in_(column_ids),
            s.catalog_column.c.sample_values.is_not(None),
        )
        .values(sample_values=None, sample_values_updated_at=None)
        .returning(s.catalog_column.c.id)
    )
    return len(rows)


def clear_tagged_pii_sample_values(column_ids: list[str] | None = None) -> int:
    """Delete samples for every selected column currently carrying ``PII``."""

    pii_tagged = (
        select(1)
        .select_from(s.tag_target.join(s.tag, s.tag.c.id == s.tag_target.c.tag_id))
        .where(
            s.tag_target.c.column_id == s.catalog_column.c.id,
            func.lower(s.tag.c.name) == PII_TAG_NAME.casefold(),
        )
        .exists()
    )
    selected_ids = select(s.catalog_column.c.id).where(pii_tagged)
    if column_ids is not None:
        if not column_ids:
            return 0
        selected_ids = selected_ids.where(s.catalog_column.c.id.in_(column_ids))
    tagged_ids = [row["id"] for row in store().query_read(selected_ids)]
    delete_column_sample_embeddings(tagged_ids)

    statement = (
        s.catalog_column.update()
        .where(s.catalog_column.c.sample_values.is_not(None), pii_tagged)
        .values(sample_values=None, sample_values_updated_at=None)
        .returning(s.catalog_column.c.id)
    )
    if column_ids is not None:
        statement = statement.where(s.catalog_column.c.id.in_(column_ids))
    return len(store().query_write(statement))


def purge_expired_sample_values(*, retention_days: int | None = None) -> int:
    """Delete samples older than the configured retention period.

    A non-null sample without a timestamp predates retention tracking and is
    expired immediately rather than retained indefinitely.
    """

    days = sample_retention_days() if retention_days is None else retention_days
    days = min(max(days, 1), _MAX_SAMPLE_RETENTION_DAYS)
    cutoff = datetime.now(UTC) - timedelta(days=days)
    expired = [
        row["id"]
        for row in store().query_read(
            select(s.catalog_column.c.id).where(
                s.catalog_column.c.sample_values.is_not(None),
                or_(
                    s.catalog_column.c.sample_values_updated_at.is_(None),
                    s.catalog_column.c.sample_values_updated_at < cutoff,
                ),
            )
        )
    ]
    delete_column_sample_embeddings(expired)
    rows = store().query_write(
        s.catalog_column.update()
        .where(
            s.catalog_column.c.sample_values.is_not(None),
            or_(
                s.catalog_column.c.sample_values_updated_at.is_(None),
                s.catalog_column.c.sample_values_updated_at < cutoff,
            ),
        )
        .values(sample_values=None, sample_values_updated_at=None)
        .returning(s.catalog_column.c.id)
    )
    return len(rows)


def is_column_safe_for_data_movement(
    *,
    database_name: str,
    table_name: str,
    column_name: str,
    schema_name: str | None = None,
) -> bool:
    """Whether a catalog column was classified and is not tagged as PII.

    Missing and ambiguous catalog references fail closed. Live probes must not
    move values from a column whose policy cannot be established.
    """

    pii_tagged = (
        select(1)
        .select_from(s.tag_target.join(s.tag, s.tag.c.id == s.tag_target.c.tag_id))
        .where(
            s.tag_target.c.column_id == s.catalog_column.c.id,
            func.lower(s.tag.c.name) == PII_TAG_NAME.casefold(),
        )
        .exists()
    )
    conditions = [
        func.lower(s.catalog_database.c.name) == database_name.casefold(),
        func.lower(s.catalog_table.c.name) == table_name.casefold(),
        func.lower(s.catalog_column.c.name) == column_name.casefold(),
    ]
    if schema_name:
        conditions.append(func.lower(s.catalog_schema.c.name) == schema_name.casefold())
    rows = store().query_read(
        select(
            s.catalog_column.c.pii_processed,
            pii_tagged.label("is_pii"),
        )
        .select_from(
            s.catalog_column.join(
                s.catalog_table,
                s.catalog_table.c.id == s.catalog_column.c.table_id,
            )
            .join(
                s.catalog_schema,
                s.catalog_schema.c.id == s.catalog_table.c.schema_id,
            )
            .join(
                s.catalog_database,
                s.catalog_database.c.id == s.catalog_schema.c.database_id,
            )
        )
        .where(*conditions)
        .limit(2)
    )
    return (
        len(rows) == 1
        and bool(rows[0]["pii_processed"])
        and not bool(rows[0]["is_pii"])
    )


def is_column_id_safe_for_data_movement(column_id: str) -> bool:
    """ID-based form used before accepting manually supplied sample values."""

    pii_tagged = (
        select(1)
        .select_from(s.tag_target.join(s.tag, s.tag.c.id == s.tag_target.c.tag_id))
        .where(
            s.tag_target.c.column_id == s.catalog_column.c.id,
            func.lower(s.tag.c.name) == PII_TAG_NAME.casefold(),
        )
        .exists()
    )
    rows = store().query_read(
        select(
            s.catalog_column.c.pii_processed,
            pii_tagged.label("is_pii"),
        ).where(s.catalog_column.c.id == column_id)
    )
    return (
        len(rows) == 1
        and bool(rows[0]["pii_processed"])
        and not bool(rows[0]["is_pii"])
    )
