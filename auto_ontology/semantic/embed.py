# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inline per-Term embedder used during `visit_enter`."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from auto_ontology.utils.embedding import EmbedParams, batch_embed
from auto_ontology.utils.model_config import resolve
from auto_ontology.utils.sample_values import stringify_sample_values
from auto_ontology.vdb import get_semantic_vdb
from auto_ontology.vdb.postgres import PostgresVDB
from auto_ontology.vdb.records import write_embedded_rows

logger = logging.getLogger(__name__)

# Max length of one already-formatted sample_values list entry kept in a
# non-JSON ColumnAttribute's embedding text. Historically the only cutoff
# here — kept as-is for every ordinary column so this change stays scoped to
# JSONB, not a blanket loosening.
_MAX_EMBEDDED_SAMPLE_LEN = 30

# Same idea, but for JSONB columns specifically: their sample_values entries
# are visit_enter.py's formatted `key [e.g. 'value']` strings (brackets and
# quotes included), which need more headroom than a bare value does — using
# the plain 30-char cutoff on those would routinely drop the entry whole,
# including the key name itself. Scoped to JSON-typed columns only (via the
# ``data_type`` param below) so ordinary columns are unaffected.
_MAX_EMBEDDED_JSON_SAMPLE_LEN = 60


@dataclass
class SemanticEmbedder:
    """Embeds one Term + its ColumnAttributes into the semantic VDB per call."""

    database_name: str
    embed_params: EmbedParams
    vdb: PostgresVDB

    def embed_term(
        self,
        term: dict[str, Any],
        attrs: list[dict[str, Any]],
        *,
        embed_attrs: bool = True,
    ) -> int:
        """Embed and ingest one Term and its attribute rows in a single batch.

        ``term`` entries: ``{"name", "description", "id"}``.
        ``attrs`` entries include ``name``, ``term_name``, ``source_column``,
        ``description``, ``id``, and the owning table metadata.
        ``id`` is the catalog ``id`` (UUID) and, when present, lands
        in the embedded row's metadata.
        The Term's text lists every attribute in *attrs*; with ``embed_attrs``
        false only the Term row is written.
        Returns the number of rows actually written to the VDB.
        """
        rows = _build_rows(self.database_name, term, attrs)
        if not embed_attrs:
            rows = [r for r in rows if r["metadata"]["label"] == "Term"]
        if not rows:
            return 0

        embedded_df = batch_embed(rows, self.embed_params)
        if embedded_df is None or embedded_df.empty:
            logger.warning(
                "Inline embed produced no rows for Term %s", term.get("name")
            )
            return 0

        with_embeddings = [
            row
            for row in embedded_df.to_dict(orient="records")
            if (row.get("metadata") or {}).get("embedding")
        ]
        if not with_embeddings:
            logger.warning(
                "Inline embed produced 0/%d rows with embeddings for Term %s",
                len(rows),
                term.get("name"),
            )
            return 0

        write_embedded_rows(self.vdb, with_embeddings)
        return len(with_embeddings)

    def embed_column_attributes(self, attrs: list[dict[str, Any]]) -> int:
        """Embed only ColumnAttribute rows, without rewriting the parent Term row."""
        return self.embed_term({}, attrs)


def build_semantic_embedder(
    database_name: str,
    *,
    reset: bool,
) -> SemanticEmbedder | None:
    """Construct an embedder bound to the semantic VDB, or None when disabled."""
    api_key = resolve("EMBED", "API_KEY")
    if not api_key:
        logger.warning(
            "EMBED_API_KEY / DEFAULT_MODELS_API_KEY not set — "
            "semantic VDB embedding disabled"
        )
        return None

    embed_params = EmbedParams(
        embed_invoke_url=resolve("EMBED", "ENDPOINT"),
        model_name=resolve("EMBED", "MODEL"),
        api_key=api_key,
    )
    vdb = get_semantic_vdb(database_name=database_name, reset=reset)
    return SemanticEmbedder(
        database_name=database_name,
        embed_params=embed_params,
        vdb=vdb,
    )


def embed_all_semantic_nodes(
    embedder: SemanticEmbedder,
) -> int:
    """Embed every Term + ColumnAttribute currently in the store in a single batch.

    Fetches all semantic nodes, builds one combined DataFrame, makes a single
    HTTP call to the embed endpoint, and writes all results to the VDB in one
    pass.  Returns the total number of rows written.
    """
    from collections import defaultdict

    from auto_ontology.dal.terms import fetch_all_terms_and_attributes

    terms, attrs = fetch_all_terms_and_attributes()
    if not terms and not attrs:
        logger.info("embed_all_semantic_nodes: nothing to embed")
        return 0

    attrs_by_term: dict[str, list[dict]] = defaultdict(list)
    for attr in attrs:
        term_name = attr.get("term_name") or ""
        if term_name:
            attrs_by_term[term_name].append(attr)

    all_rows: list[dict] = []
    for term in terms:
        term_name = term.get("name") or ""
        if not term_name:
            continue
        all_rows.extend(
            _build_rows(
                embedder.database_name,
                {
                    "name": term_name,
                    "description": term.get("description") or "",
                    "synonyms": term.get("synonyms") or [],
                    "id": term.get("id"),
                    "schema_names": term.get("schema_names") or [],
                },
                attrs_by_term.get(term_name, []),
            )
        )

    if not all_rows:
        logger.info("embed_all_semantic_nodes: no rows to embed")
        return 0

    embedded_df = batch_embed(all_rows, embedder.embed_params)
    if embedded_df is None or embedded_df.empty:
        logger.warning("embed_all_semantic_nodes: embed produced no rows")
        return 0

    with_embeddings = [
        row
        for row in embedded_df.to_dict(orient="records")
        if (row.get("metadata") or {}).get("embedding")
    ]
    if not with_embeddings:
        logger.warning(
            "embed_all_semantic_nodes: 0/%d rows had embeddings", len(all_rows)
        )
        return 0

    write_embedded_rows(embedder.vdb, with_embeddings)
    logger.info("embed_all_semantic_nodes: %d VDB row(s) written", len(with_embeddings))
    return len(with_embeddings)


def _sample_values(raw: Any, data_type: str | None = None) -> list[str]:
    """The sample values short enough to embed.

    *data_type* is the owning column's declared type. JSON-typed columns get a
    higher per-entry length cutoff (see ``_MAX_EMBEDDED_JSON_SAMPLE_LEN``)
    since their sample_values entries are visit_enter.py's formatted
    ``key [e.g. 'value']`` strings, not bare values — every other column keeps
    the original cutoff unchanged.
    """
    max_len = (
        _MAX_EMBEDDED_JSON_SAMPLE_LEN
        if "json" in (data_type or "").lower()
        else _MAX_EMBEDDED_SAMPLE_LEN
    )
    return stringify_sample_values(raw, max_len=max_len) or []


# Ingestion appends the column's samples and storage notation to an attribute's
# description (see ``semantic.deterministic``). The embedding states both once,
# from the column, so they are cut from the description here.
_SAMPLES_SUFFIX_RE = re.compile(r"\s*—\s*samples:.*?(?=\s*—\s*format:|$)", re.S)
_FORMAT_CLAUSE_RE = re.compile(r"(?:^|\s*—\s*)format:\s*(.*?)\s*$", re.S)


def _split_description(text: str | None) -> tuple[str, str | None]:
    """``(description, format)`` with the appended samples and format removed."""
    text = _SAMPLES_SUFFIX_RE.sub("", (text or "").strip())
    notation = None
    if match := _FORMAT_CLAUSE_RE.search(text):
        notation = match.group(1) or None
        text = text[: match.start()]
    return text.strip(), notation


def _same_text(a: str, b: str) -> bool:
    def norm(t: str) -> str:
        return re.sub(r"\W+", " ", t).strip().lower()

    return norm(a) == norm(b)


def _sentence(text: str) -> str:
    text = text.strip()
    return text if not text or text[-1] in ".!?" else f"{text}."


def _yes_no(value: Any) -> str | None:
    return None if value is None else ("yes" if value else "no")


def _term_text(
    term: dict[str, Any], attrs: list[dict[str, Any]], database_name: str
) -> str:
    """A Term first, then its table in the shape of the table's own embedding.

    ``term, synonyms, term_description, database_name, schema_name,
    table_name, table_description, attributes: {name, column, data_type,
    description}``. A Term over several tables lists them as
    ``tables: {table, table_description}`` and names each attribute's table. Sample values are left to the attribute rows, so
    profiling a column does not stale its Term.
    """
    members = [a for a in attrs if a.get("name")]
    tables: dict[str, dict[str, Any]] = {}
    for a in members:
        if a.get("table_name"):
            tables.setdefault(_qualified_table(a), a)

    parts = [f"term: {term['name']}"]
    if synonyms := [s for s in term.get("synonyms") or [] if s]:
        parts.append(f"synonyms: {', '.join(synonyms)}")
    if term.get("description"):
        parts.append(f"term_description: {term['description'].strip()}")
    database = next(
        (a["database_name"] for a in members if a.get("database_name")), database_name
    )
    if database:
        parts.append(f"database_name: {database}")

    if len(tables) == 1:
        (table,) = tables.values()
        parts += [
            f"schema_name: {table.get('schema_name') or ''}",
            f"table_name: {table['table_name']}",
        ]
        parts += [f"{k}: {v}" for k, v in _table_facts(table).items()]
    elif tables:
        pieces = [
            _braces({"table": name, **_table_facts(table)})
            for name, table in tables.items()
        ]
        parts.append(f"tables: {','.join(pieces)}")

    pieces = []
    for a in members:
        description = (
            _split_description(a.get("description"))[0]
            or (_split_description(a.get("column_description"))[0])
        )
        pieces.append(
            _braces(
                {
                    "name": a["name"],
                    "table": _qualified_table(a) if len(tables) > 1 else None,
                    "column": a.get("column_name") or a.get("source_column"),
                    "data_type": a.get("column_data_type") or a.get("datatype"),
                    "description": description,
                }
            )
        )
    if pieces:
        parts.append(f"attributes: {','.join(pieces)}")
    return ", ".join(parts)


def _qualified_table(a: dict[str, Any]) -> str:
    return ".".join(p for p in (a.get("schema_name"), a.get("table_name")) if p)


def _table_facts(a: dict[str, Any]) -> dict[str, str]:
    description = (a.get("table_description") or "").strip()
    return {"table_description": description} if description else {}


def _braces(fields: dict[str, Any]) -> str:
    return "{" + ", ".join(f"{k}: {v}" for k, v in fields.items() if v) + "}"


def _attribute_text(
    a: dict[str, Any], owner: str, synonyms: list[str], database_name: str
) -> str:
    """An attribute in words, then everything the owning column's own text has.

    ``Order Date is an attribute of Purchase Order (also: PO). It represents
    column ord_dt of table sales.orders in database shop. Description: … data
    type: date, format: YYYY-MM-DD, unique: no, sample values: …``
    """
    text = f"{a['name']} is an attribute of {owner}" if owner else a["name"]
    if synonyms:
        text += f" (also: {', '.join(synonyms)})"
    text += "."

    column = a.get("column_name") or a.get("source_column")
    if column:
        table = ".".join(p for p in (a.get("schema_name"), a.get("table_name")) if p)
        where = f" of table {table}" if table else ""
        database = a.get("database_name") or database_name
        where += f" in database {database}" if database else ""
        text += f" It represents column {column}{where}."

    description, notation = _split_description(a.get("description"))
    column_description, _ = _split_description(a.get("column_description"))
    if description:
        text += f" Description: {_sentence(description)}"
    if column_description and not _same_text(column_description, description):
        text += f" Column description: {_sentence(column_description)}"

    data_type = a.get("column_data_type") or a.get("datatype")
    samples = _sample_values(a.get("sample_values"), data_type)
    pk = {str(c).lower() for c in a.get("table_pk") or []}
    facts = {
        "data type": data_type,
        "format": a.get("format") or notation,
        "primary key": "yes" if column and column.lower() in pk else None,
        "unique": _yes_no(a.get("is_unique")),
        "sample values": ", ".join(samples),
    }
    details = ", ".join(f"{k}: {v}" for k, v in facts.items() if v)
    return f"{text} {_sentence(details)}" if details else text


def _build_rows(
    database_name: str,
    term: dict[str, Any],
    attrs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    term_name = term.get("name")
    if term_name:
        text = _term_text(term, attrs, database_name)
        path = f"semantic:term:{term_name}"
        fields: dict[str, Any] = {
            "label": "Term",
            "name": term_name,
            "database_name": database_name,
            "source_path": path,
        }
        if term.get("id"):
            fields["id"] = term["id"]
        schema_names = [s for s in (term.get("schema_names") or []) if s]
        if schema_names:
            fields["schema_names"] = schema_names
        rows.append(
            {
                "text": text,
                "path": path,
                "page_number": -1,
                "metadata": {**fields, "content_metadata": dict(fields)},
            }
        )

    term_synonyms: list[str] = term.get("synonyms") or []

    for a in attrs:
        attr_name = a.get("name")
        if not attr_name:
            continue
        owner = a.get("term_name") or term_name or ""
        synonyms = [s for s in (a.get("term_synonyms") or term_synonyms) if s]
        text = _attribute_text(a, owner, synonyms, database_name)
        path = (
            f"semantic:attr:{a['id']}"
            if a.get("id")
            else f"semantic:attr:{owner}:{a.get('source_column')}"
        )
        fields: dict[str, Any] = {
            "label": "ColumnAttribute",
            "name": attr_name,
            "term_name": owner,
            "source_column": a.get("source_column"),
            "table_id": a.get("table_id"),
            "table_name": a.get("table_name"),
            # Lets semantic FK resolution filter VDB candidates down to
            # columns that can validly serve as a referenced key — see
            # semantic_fk._resolve_via_vdb.
            "is_unique": a.get("is_unique"),
            "database_name": database_name,
            "source_path": path,
        }
        if a.get("schema_name"):
            fields["schema_name"] = a["schema_name"]
        if data_type := a.get("datatype") or a.get("column_data_type"):
            # Threaded through so downstream consumers (e.g. entity_resolution's
            # composite-column check) can read the real column type instead of
            # guessing from description text. Only present on rows embedded after
            # this field was added — existing embedded rows keep the text-based
            # fallback until they're re-embedded.
            fields["data_type"] = data_type
        if a.get("id"):
            fields["id"] = a["id"]
        rows.append(
            {
                "text": text,
                "path": path,
                "page_number": -1,
                "metadata": {**fields, "content_metadata": dict(fields)},
            }
        )

    return rows
