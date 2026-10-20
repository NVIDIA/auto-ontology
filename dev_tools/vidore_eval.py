# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score chunk retrieval against the ViDoRe v3 HR qrels.

Answers one question: does embedding chunks as text alone, rather than as the
text-plus-image pairs the NeMo Retriever graph produces today, cost retrieval
quality? Both arms are scored here against the same judgements, so the delta
between them is the answer.

The corpus is not in the repository (``unstructured_ao/vidore_v3_hr/``, which
.gitignore excludes): 14 EU publications, 318 queries and 1,731 page-level
graded judgements.

Usage::

    # 0. ingest the corpus first -- this is the slow step
    uv run --no-sync python -m unstructured_ao.ingest \
        unstructured_ao/vidore_v3_hr/pdfs

    # 1. prove the qrels join before believing any score
    uv run --no-sync python -m dev_tools.vidore_eval validate

    # 2. score an arm
    uv run --no-sync python -m dev_tools.vidore_eval score --out arm_a.json

Scoring never writes to the database, so a bug here costs a re-run of step 2
rather than a re-ingest.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("dev_tools.vidore_eval")

#: Where the untracked corpus lives, relative to the repo root.
CORPUS = Path(__file__).resolve().parents[1] / "unstructured_ao" / "vidore_v3_hr"

#: Cutoffs to report. 5 is ViDoRe's headline; 10 shows whether a difference is
#: a ranking change or a reachability one.
CUTOFFS = (5, 10)

#: Chunks to pull before collapsing to pages. Chunks are per element, so one
#: page can produce many of them -- asking for 10 chunks can yield three pages.
#: The pool is deduplicated down to pages and then cut at k, so "top 10" means
#: ten distinct pages, which is the unit the judgements grade.
POOL = 100

#: ``{(doc_id, page): gain}``. Gain is graded 1 or 2 in these qrels.
GainsByPage = dict[tuple[str, int], int]


@dataclass
class Scores:
    """Mean metrics over the queries that were scored."""

    queries: int
    ndcg: dict[int, float]
    recall: dict[int, float]
    #: Queries whose judged pages are all missing from the index. Retrieval
    #: cannot win these, so they are reported rather than silently averaged in.
    unreachable: int = 0
    #: ``{query_id: {"ndcg@5": ...}}``. Kept because the arms differ by a few
    #: points at most, and only a paired per-query test can say whether a gap
    #: that small is real.
    per_query: dict[str, dict[str, float]] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "queries": self.queries,
            "unreachable": self.unreachable,
            "ndcg": {str(k): round(v, 4) for k, v in self.ndcg.items()},
            "recall": {str(k): round(v, 4) for k, v in self.recall.items()},
            "per_query": self.per_query,
        }


# ---------------------------------------------------------------------------
# Corpus (pure, no database)
# ---------------------------------------------------------------------------


def load_queries(corpus: Path = CORPUS) -> list[tuple[str, str]]:
    """``(query_id, query_text)`` for every row of ``queries_en.csv``."""
    path = corpus / "queries_en.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. The ViDoRe corpus is untracked -- see the "
            f"module docstring."
        )
    with path.open(newline="", encoding="utf-8") as handle:
        return [(row["query_id"], row["query"]) for row in csv.DictReader(handle)]


def load_qrels(corpus: Path = CORPUS) -> dict[str, GainsByPage]:
    """Graded judgements as ``{query_id: {(doc_id, page): gain}}``.

    ``doc_id`` carries no file extension and ``page`` is 1-based, which is the
    form :func:`retrieved_pages` normalises retrieved chunks into.
    """
    path = corpus / "qrels" / "qrels_en.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found.")
    qrels: dict[str, GainsByPage] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["doc_id"], int(row["page_number"]))
            qrels[row["query_id"]][key] = int(row["score"])
    return dict(qrels)


# ---------------------------------------------------------------------------
# Metrics (pure, no database)
# ---------------------------------------------------------------------------


def ndcg_at_k(ranked: list[tuple[str, int]], gains: GainsByPage, k: int) -> float:
    """Graded nDCG over the first *k* ranked pages.

    Linear gain discounted by ``log2(rank + 1)``, which is trec_eval's
    convention. The ideal ranking is every judged page in descending gain, cut
    at *k*, so a query with more relevant pages than *k* can still reach 1.0.
    """
    dcg = sum(
        gains.get(page, 0) / math.log2(rank + 1)
        for rank, page in enumerate(ranked[:k], start=1)
    )
    ideal = sorted(gains.values(), reverse=True)[:k]
    idcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0


def recall_at_k(ranked: list[tuple[str, int]], gains: GainsByPage, k: int) -> float:
    """Share of judged pages found in the first *k* ranked pages.

    Bounded above by ``k / len(gains)``: some queries here have up to 30 judged
    pages, so their recall@10 cannot reach 1.0. It stays comparable between
    arms, which is all it is used for.
    """
    if not gains:
        return 0.0
    found = sum(1 for page in ranked[:k] if page in gains)
    return found / len(gains)


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

#: Arm A. Written by the NeMo Retriever graph, one vector per chunk, embedded
#: as text *and* the page image.
ARM_MULTIMODAL = "unstructured_chunks"

#: Arm A-prime. The same chunk text, same model, same ids -- embedded without
#: the image. Re-embedding rather than re-extracting is what makes the two
#: comparable: extraction is held fixed, so any delta is the modality.
ARM_TEXT_ONLY = "unstructured_chunks_text"


def use_chunk_table(table: str) -> None:
    """Point the chunk vector store at *table* for the rest of this process.

    ``get_chunk_store`` caches one store per process and reads the table name
    from ``config`` when it builds it, so selecting an arm means setting the
    name and dropping the cached store. Reaching into the module global is
    deliberate: the alternative is a second store implementation that could
    drift from the one ingest actually writes through.
    """
    from unstructured_ao import config, vectorstore

    config.CHUNK_TABLE = table
    vectorstore._store = None


def reembed(target: str = ARM_TEXT_ONLY, *, batch: int = 256) -> int:
    """Embed every chunk's text without its image into *target*.

    Reads ``unstructured_chunk.content`` -- the text the NeMo graph already
    extracted -- so no document is parsed twice and the two arms differ in
    exactly one way. Vectors are written under the same chunk ids, which keeps
    the join in :func:`retrieved_pages` working unchanged.
    """
    from sqlalchemy import select

    from auto_ontology.dal.schema import unstructured_chunk, unstructured_document
    from auto_ontology.dal.session import get_engine
    from auto_ontology.utils import get_embed_params
    from auto_ontology.utils.embedding import embed_texts

    statement = (
        select(
            unstructured_chunk.c.id,
            unstructured_chunk.c.content,
            unstructured_chunk.c.content_type,
            unstructured_chunk.c.chunk_index,
            unstructured_chunk.c.page_number,
            unstructured_chunk.c.document_id,
            unstructured_document.c.name,
        )
        .select_from(
            unstructured_chunk.join(
                unstructured_document,
                unstructured_document.c.id == unstructured_chunk.c.document_id,
            )
        )
        .order_by(unstructured_chunk.c.document_id, unstructured_chunk.c.chunk_index)
    )
    with get_engine().connect() as conn:
        rows = [dict(row) for row in conn.execute(statement).mappings()]
    if not rows:
        raise RuntimeError("No chunks to re-embed -- ingest Arm A first.")
    logger.info("re-embedding %d chunk(s) text-only into %s", len(rows), target)

    params = get_embed_params()
    use_chunk_table(target)
    written = 0
    store = None
    for start in range(0, len(rows), batch):
        window = rows[start : start + batch]
        vectors = embed_texts([row["content"] for row in window], params, "passage")

        texts: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict] = []
        ids: list[str] = []
        for row, vector in zip(window, vectors):
            if not vector:
                continue
            texts.append(row["content"])
            embeddings.append([float(value) for value in vector])
            metadatas.append(
                {
                    "document_id": row["document_id"],
                    "chunk_index": row["chunk_index"],
                    "content_type": row["content_type"],
                    "page_number": row["page_number"],
                    "source": row["name"],
                }
            )
            ids.append(row["id"])

        if not embeddings:
            continue
        if store is None:
            from unstructured_ao.vectorstore import get_chunk_store

            store = get_chunk_store(vector_size=len(embeddings[0]))
        store.add_embeddings(
            texts=texts, embeddings=embeddings, metadatas=metadatas, ids=ids
        )
        written += len(embeddings)
        logger.info("  %d/%d", min(start + batch, len(rows)), len(rows))

    if written < len(rows):
        logger.warning(
            "%d of %d chunk(s) came back unembedded", len(rows) - written, len(rows)
        )
    return written


# ---------------------------------------------------------------------------
# Retrieval (needs the database and the embedding endpoint)
# ---------------------------------------------------------------------------


def retrieved_pages(query: str, *, pool: int = POOL) -> list[tuple[str, int]]:
    """Distinct ``(doc_id, page)`` for *query*, best first.

    Collapses the chunk ranking onto pages: a page enters at the rank of its
    best chunk, which is the unit the judgements grade. Document names lose
    their extension to match the qrels' ``doc_id``.
    """
    from unstructured_ao import search

    pages: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for passage in search.chunk_search(query, k=pool):
        if not passage.document or passage.page is None:
            continue
        key = (Path(passage.document).stem, int(passage.page))
        if key not in seen:
            seen.add(key)
            pages.append(key)
    return pages


def score_arm(*, pool: int = POOL, limit: int | None = None) -> Scores:
    """Run every query and average the metrics."""
    queries = load_queries()
    qrels = load_qrels()
    if limit:
        queries = queries[:limit]

    indexed = indexed_pages()
    totals_ndcg: dict[int, float] = dict.fromkeys(CUTOFFS, 0.0)
    totals_recall: dict[int, float] = dict.fromkeys(CUTOFFS, 0.0)
    scored = 0
    unreachable = 0
    per_query: dict[str, dict[str, float]] = {}

    for number, (query_id, text) in enumerate(queries, start=1):
        gains = qrels.get(query_id)
        if not gains:
            continue
        if not (set(gains) & indexed):
            unreachable += 1
        ranked = retrieved_pages(text, pool=pool)
        row: dict[str, float] = {}
        for k in CUTOFFS:
            row[f"ndcg@{k}"] = ndcg_at_k(ranked, gains, k)
            row[f"recall@{k}"] = recall_at_k(ranked, gains, k)
            totals_ndcg[k] += row[f"ndcg@{k}"]
            totals_recall[k] += row[f"recall@{k}"]
        per_query[query_id] = row
        scored += 1
        if number % 25 == 0:
            logger.info("scored %d/%d queries", number, len(queries))

    if not scored:
        raise RuntimeError("No query matched the qrels -- check the corpus files.")
    return Scores(
        queries=scored,
        ndcg={k: totals_ndcg[k] / scored for k in CUTOFFS},
        recall={k: totals_recall[k] / scored for k in CUTOFFS},
        unreachable=unreachable,
        per_query=per_query,
    )


# ---------------------------------------------------------------------------
# The join gate
# ---------------------------------------------------------------------------


def indexed_pages() -> set[tuple[str, int]]:
    """``(doc_id, page)`` actually present in ``unstructured_chunk``."""
    from sqlalchemy import select

    from auto_ontology.dal.schema import unstructured_chunk, unstructured_document
    from auto_ontology.dal.session import get_engine

    statement = select(
        unstructured_document.c.name, unstructured_chunk.c.page_number
    ).select_from(
        unstructured_chunk.join(
            unstructured_document,
            unstructured_document.c.id == unstructured_chunk.c.document_id,
        )
    )
    with get_engine().connect() as conn:
        return {
            (Path(row["name"]).stem, int(row["page_number"]))
            for row in conn.execute(statement).mappings()
            if row["page_number"] is not None
        }


def validate() -> bool:
    """Report whether the qrels join holds. Returns False if it does not.

    Three ways the join breaks silently, each of which leaves both arms looking
    equally bad while the comparison still appears valid: document names that
    do not match ``doc_id``, page numbering that is off by one, and documents
    that never got ingested at all.
    """
    judged: set[tuple[str, int]] = set()
    for gains in load_qrels().values():
        judged |= set(gains)

    indexed = indexed_pages()
    if not indexed:
        print("FAIL: no chunks carry a page number. Ingest first.")
        return False

    judged_docs = {doc for doc, _ in judged}
    indexed_docs = {doc for doc, _ in indexed}

    print(f"documents judged : {len(judged_docs)}")
    print(f"documents indexed: {len(indexed_docs)}")
    if missing := judged_docs - indexed_docs:
        print(f"  judged but not indexed: {sorted(missing)}")
    if unknown := indexed_docs - judged_docs:
        print(f"  indexed but not judged: {sorted(unknown)}")

    overlap = len(judged & indexed) / len(judged)
    print(f"\njudged pages present in the index: {overlap:.1%}")

    # An off-by-one shows up as a range shifted by exactly one at the low edge.
    print("\nper-document page range (judged vs indexed):")
    shifted = []
    for doc in sorted(judged_docs):
        judged_range = sorted(page for d, page in judged if d == doc)
        indexed_range = sorted(page for d, page in indexed if d == doc)
        if not indexed_range:
            print(f"  {doc}: judged {judged_range[0]}-{judged_range[-1]}, indexed NONE")
            continue
        print(
            f"  {doc}: judged {judged_range[0]}-{judged_range[-1]}, "
            f"indexed {indexed_range[0]}-{indexed_range[-1]}"
        )
        if indexed_range[0] == 0:
            shifted.append(doc)

    if shifted:
        print(
            f"\nFAIL: {len(shifted)} document(s) start at page 0, but the qrels "
            f"are 1-based. Chunk pages need +1 before scoring."
        )
        return False
    if overlap < 0.5:
        print(
            f"\nFAIL: only {overlap:.1%} of judged pages are indexed. Expect a "
            f"name or numbering mismatch, not merely a partial ingest."
        )
        return False

    print("\nOK: the join looks sound.")
    return True


def compare(baseline: Path, other: Path, *, samples: int = 10_000) -> None:
    """Paired comparison of two score files, metric by metric.

    The arms are a few points apart at most, and they answer the same 318
    queries, so the mean difference alone cannot say whether the gap is real.
    A paired bootstrap over per-query differences can: resampling queries with
    replacement gives the interval the mean would occupy across comparable
    corpora, and an interval spanning zero means the benchmark cannot tell the
    two arms apart.
    """
    import random

    first = json.loads(baseline.read_text())
    second = json.loads(other.read_text())
    shared = sorted(set(first["per_query"]) & set(second["per_query"]))
    if not shared:
        raise RuntimeError("The two runs share no queries -- compare a matched pair.")

    print(f"{first['table']}  ->  {second['table']}")
    print(f"paired over {len(shared)} queries, {samples} bootstrap samples\n")
    print(
        f"{'metric':12} {'baseline':>9} {'other':>9} {'delta':>9} {'95% CI':>18}  verdict"
    )

    rng = random.Random(0)
    indexes = range(len(shared))
    for metric in (f"{name}@{k}" for name in ("ndcg", "recall") for k in CUTOFFS):
        diffs = [
            second["per_query"][q][metric] - first["per_query"][q][metric]
            for q in shared
        ]
        mean = sum(diffs) / len(diffs)
        means = []
        for _ in range(samples):
            picks = [diffs[rng.choice(indexes)] for _ in indexes]
            means.append(sum(picks) / len(picks))
        means.sort()
        low = means[int(0.025 * samples)]
        high = means[int(0.975 * samples)]
        verdict = "no difference" if low <= 0 <= high else "real"
        base = sum(first["per_query"][q][metric] for q in shared) / len(shared)
        now = sum(second["per_query"][q][metric] for q in shared) / len(shared)
        print(
            f"{metric:12} {base:9.4f} {now:9.4f} {mean:+9.4f} "
            f"[{low:+.4f}, {high:+.4f}]  {verdict}"
        )


def main() -> None:
    import argparse

    from auto_ontology.env import load_env

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    load_env()

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate", help="check the qrels join before scoring")

    again = sub.add_parser("reembed", help="build the text-only arm from Arm A")
    again.add_argument("--target", default=ARM_TEXT_ONLY)

    run = sub.add_parser("score", help="score one arm")
    run.add_argument(
        "--table",
        default=ARM_MULTIMODAL,
        help=f"vector table to search ({ARM_MULTIMODAL} or {ARM_TEXT_ONLY})",
    )
    run.add_argument("--pool", type=int, default=POOL)
    run.add_argument("--limit", type=int, default=None, help="first N queries only")
    run.add_argument("--out", type=Path, default=None, help="write results as JSON")

    against = sub.add_parser("compare", help="paired bootstrap over two score files")
    against.add_argument("baseline", type=Path)
    against.add_argument("other", type=Path)
    args = parser.parse_args()

    if args.command == "validate":
        raise SystemExit(0 if validate() else 1)

    if args.command == "reembed":
        print(f"wrote {reembed(args.target)} vector(s) to {args.target}")
        return

    if args.command == "compare":
        compare(args.baseline, args.other)
        return

    use_chunk_table(args.table)
    scores = score_arm(pool=args.pool, limit=args.limit)
    print(f"arm: {args.table}")
    print(json.dumps(scores.as_dict(), indent=2))
    if args.out:
        payload = {"table": args.table, **scores.as_dict()}
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
