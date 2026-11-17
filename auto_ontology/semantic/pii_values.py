# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Withholding the values of a column from the moment it is tagged ``PII``.

Compilation leaves a PII column's values out of everything it samples, but it
only visits tables without a Term, so a table compiled before its column was
tagged keeps the values in the catalog and in its ColumnAttribute vectors.
:func:`withhold_pii_column_values` closes that: it clears the column's value
profile and re-embeds the attributes built from it, whose vector text carries
``sample_values``. The Term row embeds no values, so it is left alone.

Called wherever a column can gain the tag -- after PII detection, on a hand
label, and after rules apply theirs -- and idempotent, since a column cleared
once is not touched again. Running it once over the catalog is the backfill::

    uv run python -m auto_ontology.semantic.pii_values
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from auto_ontology.dal.pii import clear_values_of_pii_columns
from auto_ontology.dal.terms import fetch_terms_and_attributes_for_table
from auto_ontology.semantic.embed import SemanticEmbedder, build_semantic_embedder

logger = logging.getLogger(__name__)


def withhold_pii_column_values(column_ids: list[str] | None = None) -> int:
    """Clear the value profile of PII-tagged columns and re-embed their attributes.

    Every tagged column still holding values, or only those among *column_ids*.
    Returns how many columns were cleared.

    The catalog is cleared before anything is embedded, so a failed re-embed
    leaves the stale vectors behind but never stored values for the next
    embed to pick up. Re-embedding is best-effort per table, and skipped with a
    warning when no embedder is configured.
    """
    cleared = clear_values_of_pii_columns(column_ids)
    if not cleared:
        return 0

    columns_by_table: dict[tuple[str | None, str], set[str]] = defaultdict(set)
    for row in cleared:
        columns_by_table[(row["database_name"], row["table_id"])].add(
            row["column_name"]
        )
    logger.info(
        "Withheld the values of %d PII column(s) across %d table(s)",
        len(cleared),
        len(columns_by_table),
    )

    embedders: dict[str, SemanticEmbedder | None] = {}
    for (database_name, table_id), column_names in columns_by_table.items():
        if database_name is None:
            continue
        if database_name not in embedders:
            embedders[database_name] = build_semantic_embedder(
                database_name, reset=False
            )
        embedder = embedders[database_name]
        if embedder is None:
            logger.warning(
                "No semantic embedder for %s — vectors of %d PII column(s) in "
                "table %s still carry their old sample values",
                database_name,
                len(column_names),
                table_id,
            )
            continue
        try:
            _reembed_attributes(embedder, table_id, column_names)
        except Exception:
            logger.warning(
                "Could not re-embed the PII columns of table %s",
                table_id,
                exc_info=True,
            )
    return len(cleared)


def _reembed_attributes(
    embedder: SemanticEmbedder, table_id: str, column_names: set[str]
) -> None:
    """Replace the vectors of the attributes *table_id* builds from *column_names*."""
    terms, attrs = fetch_terms_and_attributes_for_table(table_id)
    synonyms = {term["name"]: term.get("synonyms") or [] for term in terms}
    by_term: dict[str | None, list[dict[str, Any]]] = defaultdict(list)
    for attr in attrs:
        if attr.get("source_column") in column_names and attr.get("id"):
            by_term[attr.get("term_name")].append(attr)

    for term_name, term_attrs in by_term.items():
        for attr in term_attrs:
            embedder.vdb.delete_by_id(attr["id"])
        # Synonyms only, no Term name: the Term row is not rewritten.
        embedder.embed_term({"synonyms": synonyms.get(term_name, [])}, term_attrs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(f"Withheld the values of {withhold_pii_column_values()} PII column(s)")
