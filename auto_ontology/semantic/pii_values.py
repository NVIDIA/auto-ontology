# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Withholding the values of a column from the moment it is tagged ``PII``.

Compilation leaves a PII column's values out of everything it samples, but it
only visits tables without a Term, so a table compiled before its column was
tagged keeps the values in the catalog and in its ColumnAttribute vectors.
:func:`withhold_pii_column_values` closes both, and each half decides what is
left to do from its own store:

* the catalog: a tagged column whose value profile is not empty;
* the semantic VDB: a vector of a tagged column's attribute whose text still
  carries sample values. Never the catalog -- once it is cleared it no longer
  says which vectors were embedded from it, so a re-embed that failed would
  never be retried.

A stale vector is deleted whether or not it can be re-embedded: deleting needs
no embedding endpoint, and an attribute missing from search until its next
embed is the failure that leaks nothing. The Term row embeds no values and is
left alone.

Called wherever a column can gain the tag -- after PII detection, on a hand
label, and after rules apply theirs -- and idempotent. Running it once over
the catalog is the backfill::

    uv run python -m auto_ontology.semantic.pii_values
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

from auto_ontology.dal.pii import (
    clear_values_of_pii_columns,
    fetch_attributes_of_pii_columns,
)
from auto_ontology.dal.terms import fetch_terms_and_attributes_for_table
from auto_ontology.semantic.embed import (
    SAMPLE_VALUES_MARKER,
    SemanticEmbedder,
    build_semantic_embedder,
)
from auto_ontology.vdb import get_semantic_vdb

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PiiWithholdingResult:
    """What one pass found still holding the values of PII columns."""

    columns_cleared: int = 0
    vectors_deleted: int = 0
    vectors_reembedded: int = 0


def withhold_pii_column_values(
    column_ids: list[str] | None = None,
) -> PiiWithholdingResult:
    """Clear the values of PII-tagged columns from the catalog and the semantic VDB.

    Every tagged column, or only those among *column_ids*. The catalog is
    cleared first, so a re-embed reads no values back from it.
    """
    columns_cleared = clear_values_of_pii_columns(column_ids)

    stale_by_table: dict[tuple[str, str], set[str]] = defaultdict(set)
    attributes_by_database: dict[str, list[dict]] = defaultdict(list)
    for row in fetch_attributes_of_pii_columns(column_ids):
        attributes_by_database[row["database_name"]].append(row)

    vectors_deleted = 0
    for database_name, rows in attributes_by_database.items():
        try:
            vdb = get_semantic_vdb(database_name=database_name)
            stale = set(
                vdb.ids_with_content_containing(
                    [row["attribute_id"] for row in rows], SAMPLE_VALUES_MARKER
                )
            )
            for row in rows:
                if row["attribute_id"] in stale:
                    vectors_deleted += vdb.delete_by_id(row["attribute_id"])
                    stale_by_table[(database_name, row["table_id"])].add(
                        row["attribute_id"]
                    )
        except Exception:
            # Still found by the next pass: the vectors are what it checks.
            logger.warning(
                "Could not clear PII vectors for %s", database_name, exc_info=True
            )

    vectors_reembedded = _reembed(stale_by_table)
    result = PiiWithholdingResult(columns_cleared, vectors_deleted, vectors_reembedded)
    if columns_cleared or vectors_deleted:
        logger.info(
            "Withheld PII values: %d column(s) cleared, %d vector(s) deleted, "
            "%d re-embedded",
            result.columns_cleared,
            result.vectors_deleted,
            result.vectors_reembedded,
        )
    return result


def _reembed(stale_by_table: dict[tuple[str, str], set[str]]) -> int:
    """Re-embed the deleted attribute vectors; best-effort, per table."""
    embedders: dict[str, SemanticEmbedder | None] = {}
    written = 0
    try:
        for (database_name, table_id), attribute_ids in stale_by_table.items():
            if database_name not in embedders:
                embedders[database_name] = build_semantic_embedder(
                    database_name, reset=False
                )
            embedder = embedders[database_name]
            if embedder is None:
                logger.warning(
                    "No semantic embedder for %s — %d PII attribute(s) of table %s "
                    "are out of search until re-embedded",
                    database_name,
                    len(attribute_ids),
                    table_id,
                )
                continue
            try:
                written += _reembed_attributes(embedder, table_id, attribute_ids)
            except Exception:
                logger.warning(
                    "Could not re-embed the PII attributes of table %s — they are "
                    "out of search until re-embedded",
                    table_id,
                    exc_info=True,
                )
    finally:
        for embedder in embedders.values():
            if embedder is not None:
                embedder.close()
    return written


def _reembed_attributes(
    embedder: SemanticEmbedder, table_id: str, attribute_ids: set[str]
) -> int:
    """Write fresh vectors for the *attribute_ids* of *table_id*."""
    terms, attrs = fetch_terms_and_attributes_for_table(table_id)
    synonyms = {term["name"]: term.get("synonyms") or [] for term in terms}
    by_term: dict[str | None, list[dict]] = defaultdict(list)
    for attr in attrs:
        if attr.get("id") in attribute_ids:
            by_term[attr.get("term_name")].append(attr)

    written = 0
    for term_name, term_attrs in by_term.items():
        # Synonyms only, no Term name: the Term row is not rewritten.
        written += embedder.embed_term(
            {"synonyms": synonyms.get(term_name, [])}, term_attrs
        )
    return written


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(withhold_pii_column_values())
