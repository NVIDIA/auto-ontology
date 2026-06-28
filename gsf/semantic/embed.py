"""Inline per-Term embedder used during `visit_enter`."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from nemo_retriever.graph import Graph
from nemo_retriever.common.params.models import EmbedParams
from nemo_retriever.operators.embed.operators import _BatchEmbedActor
from nemo_retriever.operators.vdb import IngestVdbOperator

from gsf.vdb import get_semantic_vdb
from gsf.vdb.postgres import PostgresVDB

logger = logging.getLogger(__name__)


@dataclass
class SemanticEmbedder:
    """Embeds one Term + its ColumnAttributes into the semantic VDB per call."""

    database_name: str
    embed_params: EmbedParams
    vdb: PostgresVDB
    embed_graph: Graph = field(init=False)
    ingest_op: IngestVdbOperator = field(init=False)

    def __post_init__(self) -> None:
        self.embed_graph = Graph() >> _BatchEmbedActor(params=self.embed_params)
        self.ingest_op = IngestVdbOperator(vdb=self.vdb)

    def embed_term(
        self,
        term: dict[str, Any],
        attrs: list[dict[str, Any]],
    ) -> int:
        """Embed and ingest one Term and its attribute rows in a single batch.

        ``term`` entries: ``{"name", "description", "id"}``.
        ``attrs`` entries: ``{"name", "term_name", "source_column", "description", "id"}``.
        ``id`` is the Neo4j node ``id`` property (UUID) and, when present, lands
        in the embedded row's metadata.
        Returns the number of rows actually written to the VDB.
        """
        rows = _build_rows(self.database_name, term, attrs)
        if not rows:
            return 0

        results = self.embed_graph.execute(pd.DataFrame(rows))
        embedded_df = results[0] if results else None
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


def build_semantic_embedder(
    database_name: str,
    *,
    reset: bool,
) -> SemanticEmbedder | None:
    """Construct an embedder bound to the semantic VDB, or None when disabled."""
    api_key = os.environ.get("EMBED_API_KEY", "") or os.environ.get(
        "NVIDIA_API_KEY", ""
    )
    if not api_key:
        logger.warning(
            "EMBED_API_KEY / NVIDIA_API_KEY not set — semantic VDB embedding disabled"
        )
        return None

    embed_params = EmbedParams(
        embed_invoke_url=os.environ.get(
            "EMBED_ENDPOINT", "https://integrate.api.nvidia.com/v1"
        ),
        model_name=os.environ.get("EMBED_MODEL", "nvidia/llama-nemotron-embed-vl-1b-v2"),
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
    """Embed every Term + ColumnAttribute currently in Neo4j in a single batch.

    Fetches all semantic nodes, builds one combined DataFrame, makes a single
    HTTP call to the embed endpoint, and writes all results to the VDB in one
    pass.  Returns the total number of rows written.
    """
    from collections import defaultdict

    from gsf.neo4j.terms import fetch_all_terms_and_attributes

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
                },
                attrs_by_term.get(term_name, []),
            )
        )

    if not all_rows:
        logger.info("embed_all_semantic_nodes: no rows to embed")
        return 0

    results = embedder.embed_graph.execute(pd.DataFrame(all_rows))
    embedded_df = results[0] if results else None
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


def _format_sample_values(raw: str | None) -> str:
    """Return a ' Sample values: ...' suffix string, or empty string if unavailable."""
    if not raw:
        return ""
    try:
        import json

        values = json.loads(raw)
        non_null = [str(v) for v in values if v is not None and len(str(v)) <= 30]
        if not non_null:
            return ""
        return " Sample values: " + ", ".join(non_null) + "."
    except Exception:
        return ""


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
        sample_block = _format_sample_values(a.get("sample_values"))
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
        fields = {
            "label": "ColumnAttribute",
            "name": attr_name,
            "term_name": owner,
            "source_column": a.get("source_column"),
            "database_name": database_name,
            "source_path": path,
        }
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
