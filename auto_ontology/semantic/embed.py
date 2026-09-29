"""Inline per-Term embedder used during `visit_enter`."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from nemo_retriever.common.params.models import EmbedParams
from nemo_retriever.operators.vdb import IngestVdbOperator

from auto_ontology.utils.embedding import batch_embed
from auto_ontology.utils.model_config import resolve
from auto_ontology.utils.sample_values import stringify_sample_values
from auto_ontology.vdb import get_semantic_vdb
from auto_ontology.vdb.postgres import PostgresVDB

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
    ingest_op: IngestVdbOperator = field(init=False)

    def __post_init__(self) -> None:
        self.ingest_op = IngestVdbOperator(vdb=self.vdb)

    def embed_term(
        self,
        term: dict[str, Any],
        attrs: list[dict[str, Any]],
    ) -> int:
        """Embed and ingest one Term and its attribute rows in a single batch.

        ``term`` entries: ``{"name", "description", "id"}``.
        ``attrs`` entries include ``name``, ``term_name``, ``source_column``,
        ``description``, ``id``, and the owning table metadata.
        ``id`` is the catalog ``id`` (UUID) and, when present, lands
        in the embedded row's metadata.
        Returns the number of rows actually written to the VDB.
        """
        rows = _build_rows(self.database_name, term, attrs)
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

        self.ingest_op(with_embeddings)
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
        embed_modality="text",
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

    embedder.ingest_op(with_embeddings)
    logger.info("embed_all_semantic_nodes: %d VDB row(s) written", len(with_embeddings))
    return len(with_embeddings)


def _format_sample_values(raw: Any, data_type: str | None = None) -> str:
    """Return a ' Sample values: ...' suffix string, or empty string if unavailable.

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
    values = stringify_sample_values(raw, max_len=max_len)
    if not values:
        return ""
    return " Sample values: " + ", ".join(values) + "."


def _build_rows(
    database_name: str,
    term: dict[str, Any],
    attrs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    term_name = term.get("name")
    if term_name:
        text = f"Term: {term_name}. {term.get('description') or ''}".strip()
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
                "_embed_modality": "text",
                "path": path,
                "page_number": -1,
                "metadata": {**fields, "content_metadata": dict(fields)},
            }
        )

    term_synonyms: list[str] = term.get("synonyms") or []
    synonym_suffix = f" ({', '.join(term_synonyms)})" if term_synonyms else ""

    for a in attrs:
        attr_name = a.get("name")
        if not attr_name:
            continue
        owner = a.get("term_name") or term_name or ""
        sample_block = _format_sample_values(a.get("sample_values"), a.get("datatype"))
        text = (
            f"ColumnAttribute: {attr_name} of Term {owner}{synonym_suffix}. "
            f"{a.get('description') or ''}"
            f"{sample_block}"
        ).strip()
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
        if a.get("datatype"):
            # Threaded through so downstream consumers (e.g. entity_resolution's
            # composite-column check) can read the real column type instead of
            # guessing from description text. Only present on rows embedded after
            # this field was added — existing embedded rows keep the text-based
            # fallback until they're re-embedded.
            fields["data_type"] = a["datatype"]
        if a.get("id"):
            fields["id"] = a["id"]
        rows.append(
            {
                "text": text,
                "_embed_modality": "text",
                "path": path,
                "page_number": -1,
                "metadata": {**fields, "content_metadata": dict(fields)},
            }
        )

    return rows
