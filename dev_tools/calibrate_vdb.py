# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare the vector-store search modes on labelled entities, and calibrate their gates.

Builds the search corpus from the seeded fixture -- the same Table/Column rows
ingest writes for pagila and chinook, and the semantic layer's Term and
ColumnAttribute rows -- writes it into throwaway collections on each backend,
and searches it with every entity in ``fixtures/calibration_entities.yaml``
through :func:`~auto_ontology.retrieval.data_access.semantic_search.search_semantic_index`,
the function the app searches with. Three modes:

* ``postgres`` -- pgvector, the app embeds;
* ``es-semantic`` -- Elasticsearch ``semantic_text`` kNN;
* ``es-hybrid`` -- Elasticsearch ES|QL FORK/FUSE of BM25 and kNN.

The report has three parts:

* **Ranking** -- recall@1/3/k and MRR over the entities that have a right
  answer, overall and per kind (paraphrase / identifier). Where hybrid helps
  and where it costs, in one table.
* **Parity** -- whether the two semantic modes agree, as they must: same
  model, same rows, so the same neighbours at the same distances.
* **Gate sweep** -- for each mode, the relevance gate swept across thresholds
  (``max_distance``, read off the vector distance in every mode). An entity is
  *accepted* when its top hit passes; accepting a wrong row or any row for an
  entity with no right answer is a false accept. The threshold with the best
  F1 is the recommendation, set against the one configured today.

The numbers are only as good as the labels and only as broad as the fixture:
260-odd rows and ~70 entities say which way to move a threshold, not where it
belongs on a production catalog.

Needs the seeded fixture (see ``fixtures/sql/README.md``), Postgres, and, for
the Elasticsearch modes, ``ES_URL`` and credentials. Embeds with ``EMBED_*``.
Every row is embedded once per backend, so a run costs ~520 embedding calls'
worth of text.

Usage::

    uv run --no-sync python -m dev_tools.calibrate_vdb --output report.md
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterator, Literal

import yaml

logger = logging.getLogger("dev_tools.calibrate_vdb")

ENTITIES_PATH = Path(__file__).parent / "fixtures" / "calibration_entities.yaml"

Layer = Literal["data", "semantic"]
Mode = Literal["postgres", "es-semantic", "es-hybrid"]

#: What each mode's gate is called and how it reads a row.
GATES: dict[str, tuple[str, Callable[[dict], float], bool]] = {
    # name, value read off the top row, higher-is-better
    "cosine": ("max_distance", lambda row: float(row.get("score", math.inf)), False),
    # How hybrid is gated: its vector branch's cosine distance.
    "vector-distance": (
        "max_distance on the vector branch",
        lambda row: (
            math.inf
            if row.get("semantic_distance") is None
            else float(row["semantic_distance"])
        ),
        False,
    ),
    # Not a gate the app uses: swept for comparison, because it is the obvious
    # alternative and the reason it was rejected -- rank-built scores cannot
    # turn away an entity with no answer -- should stay visible in the report.
    "rrf": (
        "normalised RRF (comparison only)",
        lambda row: float(row.get("rrf") or 0.0),
        True,
    ),
}
COSINE_THRESHOLDS = [round(0.30 + 0.05 * i, 2) for i in range(15)]  # 0.30-1.00
RRF_THRESHOLDS = [round(0.30 + 0.05 * i, 2) for i in range(15)]  # 0.30-1.00
#: Today's settings, marked in the sweep.
CURRENT = {"cosine": (0.65, 0.75), "vector-distance": (0.65, 0.75), "rrf": ()}


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Entity:
    entity: str
    layer: Layer
    label: str
    database: str
    kind: str
    expect: tuple[str, ...]

    @property
    def has_answer(self) -> bool:
        return bool(self.expect)


def load_entities(path: Path = ENTITIES_PATH) -> list[Entity]:
    raw = yaml.safe_load(path.read_text()) or []
    return [
        Entity(
            entity=str(item["entity"]),
            layer=item["layer"],
            label=str(item["label"]),
            database=str(item["database"]),
            kind=str(item["kind"]),
            expect=tuple(str(e) for e in item.get("expect") or ()),
        )
        for item in raw
    ]


def resolve_targets(entities: list[Entity]) -> dict[str, set[str]]:
    """Each entity's expected targets as catalog ids, keyed by entity text.

    Raises ``LookupError`` naming every target that is not in the catalog, so
    a typo in the labels fails loudly instead of scoring as a miss.
    """
    from auto_ontology.dal.datasources import fetch_schemas_by_ids
    from auto_ontology.dal.terms import fetch_all_terms_and_attributes

    columns: dict[tuple[str, str], str] = {}
    for row in fetch_schemas_by_ids():
        key = f"{row['table_schema']}.{row['table_name']}.{row['column_name']}"
        columns[(row["database_name"], key)] = str(row["column_id"])
    _, attrs = fetch_all_terms_and_attributes()
    attributes = {str(a["name"]): str(a["id"]) for a in attrs if a.get("id")}

    resolved: dict[str, set[str]] = {}
    missing: list[str] = []
    for entity in entities:
        ids: set[str] = set()
        for target in entity.expect:
            found = (
                columns.get((entity.database, target))
                if entity.layer == "data"
                else attributes.get(target)
            )
            if found is None:
                missing.append(f"{entity.entity!r} -> {entity.database}:{target}")
            else:
                ids.add(found)
        resolved[entity.entity] = ids
    if missing:
        raise LookupError("Unknown calibration targets:\n  " + "\n  ".join(missing))
    return resolved


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


def build_corpus() -> dict[Layer, list[dict[str, Any]]]:
    """The rows each layer's collection holds, read from the seeded fixture.

    Data layer: what ingest writes, via the same ``CatalogEmbeddingRowsOp``.
    Semantic layer: what ``embed_all_semantic_nodes`` writes, per Term.
    Read-only; nothing in the catalog changes.
    """
    from collections import defaultdict

    from auto_ontology.dal.datasources import (
        fetch_schemas_by_ids,
        fetch_tables_and_columns_by_node_ids,
    )
    from auto_ontology.dal.terms import fetch_all_terms_and_attributes
    from auto_ontology.semantic.embed import _build_rows
    from auto_ontology.utils.embedding_rows import CatalogEmbeddingRowsOp

    table_ids: dict[str, set[str]] = defaultdict(set)
    for row in fetch_schemas_by_ids():
        table_ids[row["database_name"]].add(str(row["table_id"]))
    data_rows: list[dict[str, Any]] = []
    for database, ids in sorted(table_ids.items()):
        tables_df, columns_df, _ = fetch_tables_and_columns_by_node_ids(sorted(ids))
        frame = CatalogEmbeddingRowsOp(database_name=database).process(
            (tables_df, columns_df)
        )
        data_rows.extend(frame.to_dict(orient="records"))

    terms, attrs = fetch_all_terms_and_attributes()
    attrs_by_term: dict[str, list[dict]] = defaultdict(list)
    for attr in attrs:
        attrs_by_term[attr.get("term_name") or ""].append(attr)
    semantic_rows: list[dict[str, Any]] = []
    for term in terms:
        semantic_rows.extend(
            _build_rows(
                _term_database(term),
                {
                    "name": term.get("name"),
                    "description": term.get("description") or "",
                    "synonyms": term.get("synonyms") or [],
                    "id": term.get("id"),
                    "schema_names": term.get("schema_names") or [],
                },
                attrs_by_term.get(term.get("name") or "", []),
            )
        )
    return {"data": data_rows, "semantic": semantic_rows}


def _term_database(term: dict[str, Any]) -> str:
    """The database a fixture Term belongs to.

    ``embed_all_semantic_nodes`` is run per database and stamps that database
    on every row; the fixture's Terms span two, so each takes its owning
    table's database from the fixture definition.
    """
    from dev_tools.fixtures.seed_graph_fixture import TERMS

    owners = {name: database for name, _, (database, _, _), _ in TERMS}
    return str(term.get("database_name") or owners.get(term.get("name"), ""))


# ---------------------------------------------------------------------------
# Stores
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _backend(name: str) -> Iterator[None]:
    """``VDB_BACKEND`` for the duration, which decides who embeds on write."""
    previous = os.environ.get("VDB_BACKEND")
    os.environ["VDB_BACKEND"] = name
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("VDB_BACKEND", None)
        else:
            os.environ["VDB_BACKEND"] = previous


@dataclass
class Run:
    """Where to write and how to embed; the defaults come from the environment."""

    embed_params: Any = None  # EmbedParams for the app side
    es_endpoint: Any = None  # EmbedEndpoint for the Elasticsearch side
    backends: tuple[str, ...] = ("postgres", "elasticsearch")
    top_k: int = 10
    keep: bool = False
    suffix: str = field(default_factory=lambda: uuid.uuid4().hex[:6])


def _postgres_stores(run: Run, corpus: dict) -> Iterator[dict[Layer, Any]]:
    import psycopg

    from auto_ontology.infra.postgres import get_postgres_connection_string
    from auto_ontology.utils.embedding import batch_embed
    from auto_ontology.vdb.postgres import PostgresVDB
    from auto_ontology.vdb.records import write_embedded_rows

    stores: dict[Layer, PostgresVDB] = {}
    try:
        with _backend("postgres"):
            for layer, rows in corpus.items():
                store = PostgresVDB(
                    connection_string=get_postgres_connection_string(),
                    collection_name=f"calibration_{layer}_{run.suffix}",
                    schema_name="vdb",
                )
                stores[layer] = store
                written = write_embedded_rows(
                    store, batch_embed(rows, run.embed_params)
                )
                logger.info("postgres %s: %d/%d rows", layer, written, len(rows))
        yield stores
    finally:
        for store in stores.values():
            store.close()
        if not run.keep:
            with psycopg.connect(get_postgres_connection_string()) as conn:
                with conn.cursor() as cur:
                    for store in stores.values():
                        cur.execute(f"DROP TABLE IF EXISTS vdb.{store.collection_name}")
                conn.commit()


def _elasticsearch_stores(run: Run, corpus: dict) -> Iterator[dict[Layer, Any]]:
    from auto_ontology.infra.elasticsearch import (
        get_elasticsearch_client,
        get_elasticsearch_config,
    )
    from auto_ontology.vdb.elasticsearch import ElasticsearchVDB
    from auto_ontology.vdb.records import write_embedded_rows

    config = get_elasticsearch_config()
    client = get_elasticsearch_client(config)
    stores: dict[Layer, ElasticsearchVDB] = {}
    try:
        with _backend("elasticsearch"):
            for layer, rows in corpus.items():
                store = ElasticsearchVDB(
                    client=client,
                    index_name=f"{config.index_prefix}-calibration-{layer}-{run.suffix}",
                    endpoint=run.es_endpoint,
                )
                stores[layer] = store
                written = write_embedded_rows(store, rows)
                logger.info("elasticsearch %s: %d/%d rows", layer, written, len(rows))
        yield stores
    finally:
        if not run.keep:
            for store in stores.values():
                client.options(ignore_status=404).indices.delete(index=store.index_name)
            client.options(ignore_status=404).inference.delete(
                inference_id=run.es_endpoint.inference_id, force=True
            )


# ---------------------------------------------------------------------------
# Search and metrics
# ---------------------------------------------------------------------------


def search(store: Any, run: Run, entity: Entity) -> list[dict]:
    """*entity*'s rows, as the app would retrieve them."""
    from auto_ontology.retrieval.data_access.semantic_search import (
        search_semantic_index,
    )
    from auto_ontology.utils.retriever import Retriever

    return search_semantic_index(
        Retriever(store, run.embed_params),
        entity.entity,
        label_filter=[entity.label],
        per_label_k=run.top_k,
        database_name=entity.database,
    )


def first_correct_rank(rows: list[dict], expected: set[str]) -> int | None:
    """1-based rank of the first row whose id is expected, or ``None``."""
    for rank, row in enumerate(rows, start=1):
        if str(row.get("id")) in expected:
            return rank
    return None


def ranking_metrics(ranks: list[int | None], top_k: int) -> dict[str, float]:
    """recall@1/3/k and MRR over entities that have a right answer."""
    if not ranks:
        return {"n": 0}
    return {
        "n": len(ranks),
        "recall@1": mean(r is not None and r <= 1 for r in ranks),
        "recall@3": mean(r is not None and r <= 3 for r in ranks),
        f"recall@{top_k}": mean(r is not None for r in ranks),
        "mrr": mean(1 / r if r else 0.0 for r in ranks),
    }


@dataclass(frozen=True)
class GatePoint:
    threshold: float
    correct: int  # entities with an answer, top row right, accepted
    wrong: int  # entities with an answer, top row wrong, accepted
    negatives: int  # entities with no answer, accepted
    positives_total: int
    negatives_total: int

    @property
    def accepted(self) -> int:
        return self.correct + self.wrong + self.negatives

    @property
    def precision(self) -> float:
        return self.correct / self.accepted if self.accepted else 1.0

    @property
    def recall(self) -> float:
        return self.correct / self.positives_total if self.positives_total else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def negative_accept_rate(self) -> float:
        return self.negatives / self.negatives_total if self.negatives_total else 0.0


def gate_sweep(
    outcomes: list[tuple[Entity, list[dict], set[str]]],
    gate: str,
    thresholds: list[float],
) -> list[GatePoint]:
    """The gate at each threshold, applied to each entity's top row."""
    _, read, higher_is_better = GATES[gate]
    points = []
    for threshold in thresholds:
        correct = wrong = negatives = 0
        for entity, rows, expected in outcomes:
            if not rows:
                continue
            value = read(rows[0])
            accepted = value >= threshold if higher_is_better else value <= threshold
            if not accepted:
                continue
            if not entity.has_answer:
                negatives += 1
            elif str(rows[0].get("id")) in expected:
                correct += 1
            else:
                wrong += 1
        points.append(
            GatePoint(
                threshold=threshold,
                correct=correct,
                wrong=wrong,
                negatives=negatives,
                positives_total=sum(e.has_answer for e, _, _ in outcomes),
                negatives_total=sum(not e.has_answer for e, _, _ in outcomes),
            )
        )
    return points


def best_point(points: list[GatePoint], higher_is_better: bool) -> GatePoint:
    """Highest F1; on a tie, the stricter threshold (fewer false accepts)."""
    return max(
        points,
        key=lambda p: (p.f1, p.threshold if higher_is_better else -p.threshold),
    )


def parity(a: dict[str, list[dict]], b: dict[str, list[dict]]) -> dict[str, float]:
    """How closely two modes agree, per entity: top-1, top-k overlap, distance."""
    top1, overlap, diffs = [], [], []
    for key in a:
        ids_a = [str(r.get("id")) for r in a[key]]
        ids_b = [str(r.get("id")) for r in b.get(key, [])]
        if not ids_a and not ids_b:
            continue
        top1.append(bool(ids_a and ids_b and ids_a[0] == ids_b[0]))
        union = set(ids_a) | set(ids_b)
        overlap.append(len(set(ids_a) & set(ids_b)) / len(union) if union else 1.0)
        dist_b = {str(r.get("id")): r.get("score") for r in b.get(key, [])}
        diffs.extend(
            abs(float(r["score"]) - float(dist_b[str(r.get("id"))]))
            for r in a[key]
            if str(r.get("id")) in dist_b
        )
    return {
        "top1_agreement": mean(top1) if top1 else 1.0,
        "mean_overlap": mean(overlap) if overlap else 1.0,
        "max_distance_diff": max(diffs) if diffs else 0.0,
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def calibrate(run: Run, entities: list[Entity]) -> dict[str, Any]:
    """Build, write, search, measure. Returns the report as data."""
    targets = resolve_targets(entities)
    corpus = build_corpus()
    logger.info(
        "corpus: %d data rows, %d semantic rows; %d entities",
        len(corpus["data"]),
        len(corpus["semantic"]),
        len(entities),
    )

    results: dict[Mode, dict[str, list[dict]]] = {}
    if "postgres" in run.backends:
        for stores in _postgres_stores(run, corpus):
            with _backend("postgres"):
                results["postgres"] = {
                    e.entity: search(stores[e.layer], run, e) for e in entities
                }
    if "elasticsearch" in run.backends:
        for stores in _elasticsearch_stores(run, corpus):
            with _backend("elasticsearch"):
                results["es-semantic"] = {
                    e.entity: search(stores[e.layer], run, e) for e in entities
                }
                for store in stores.values():
                    store.search_mode = "hybrid"
                results["es-hybrid"] = {
                    e.entity: search(stores[e.layer], run, e) for e in entities
                }

    report: dict[str, Any] = {
        "corpus": {layer: len(rows) for layer, rows in corpus.items()},
        "entities": {
            "total": len(entities),
            "with_answer": sum(e.has_answer for e in entities),
            "without_answer": sum(not e.has_answer for e in entities),
        },
        "top_k": run.top_k,
        "ranking": {},
        "gates": {},
        "misses": {},
    }
    for mode, by_entity in results.items():
        outcomes = [(e, by_entity[e.entity], targets[e.entity]) for e in entities]
        answered = [(e, rows, t) for e, rows, t in outcomes if e.has_answer]
        ranking = {
            "all": ranking_metrics(
                [first_correct_rank(rows, t) for _, rows, t in answered], run.top_k
            )
        }
        for kind in sorted({e.kind for e, _, _ in answered}):
            ranking[kind] = ranking_metrics(
                [
                    first_correct_rank(rows, t)
                    for e, rows, t in answered
                    if e.kind == kind
                ],
                run.top_k,
            )
        report["ranking"][mode] = ranking

        gates = ["vector-distance", "rrf"] if mode == "es-hybrid" else ["cosine"]
        for gate in gates:
            thresholds = RRF_THRESHOLDS if gate == "rrf" else COSINE_THRESHOLDS
            points = gate_sweep(outcomes, gate, thresholds)
            best = best_point(points, GATES[gate][2])
            key = mode if gate in ("vector-distance", "cosine") else f"{mode}, {gate}"
            report["gates"][key] = {
                "gate": gate,
                "setting": GATES[gate][0],
                "current": list(CURRENT[gate]),
                "best": _point(best),
                "sweep": [_point(p) for p in points],
            }
        report["misses"][mode] = [
            {
                "entity": e.entity,
                "kind": e.kind,
                "top": rows[0].get("name") if rows else None,
                "rank": first_correct_rank(rows, t),
            }
            for e, rows, t in answered
            if first_correct_rank(rows, t) != 1
        ]
    if "postgres" in results and "es-semantic" in results:
        report["parity"] = parity(results["postgres"], results["es-semantic"])
    return report


def _point(p: GatePoint) -> dict[str, Any]:
    return {
        **asdict(p),
        "precision": round(p.precision, 3),
        "recall": round(p.recall, 3),
        "f1": round(p.f1, 3),
        "negative_accept_rate": round(p.negative_accept_rate, 3),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def render(report: dict[str, Any]) -> str:
    """The report as Markdown."""
    lines = ["# Vector-store calibration", ""]
    ent = report["entities"]
    lines += [
        f"Corpus: {report['corpus']['data']} data-layer rows, "
        f"{report['corpus']['semantic']} semantic-layer rows. "
        f"Entities: {ent['with_answer']} with a right answer, "
        f"{ent['without_answer']} without. Top-k: {report['top_k']}.",
        "",
        "## Ranking (entities with a right answer)",
        "",
        "| Mode | Kind | n | recall@1 | recall@3 | recall@k | MRR |",
        "|---|---|---|---|---|---|---|",
    ]
    k = report["top_k"]
    for mode, by_kind in report["ranking"].items():
        for kind, m in by_kind.items():
            if not m.get("n"):
                continue
            lines.append(
                f"| {mode} | {kind} | {m['n']} | {m['recall@1']:.2f} | "
                f"{m['recall@3']:.2f} | {m[f'recall@{k}']:.2f} | {m['mrr']:.2f} |"
            )
    if "parity" in report:
        p = report["parity"]
        lines += [
            "",
            "## Parity: postgres vs es-semantic",
            "",
            f"Top-1 agreement {p['top1_agreement']:.2f}, mean top-k overlap "
            f"{p['mean_overlap']:.2f}, largest distance difference on a shared "
            f"row {p['max_distance_diff']:.4f}.",
        ]
    lines += ["", "## Gates", ""]
    for mode, g in report["gates"].items():
        best = g["best"]
        current = ", ".join(str(c) for c in g["current"])
        lines += [
            f"### {mode} — `{g['setting']}` (today: {current}; best F1: "
            f"{best['threshold']})",
            "",
            "| threshold | accepted | correct | wrong | no-answer accepted | "
            "precision | recall | F1 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for point in g["sweep"]:
            marks = []
            if point["threshold"] in g["current"]:
                marks.append("today")
            if point["threshold"] == best["threshold"]:
                marks.append("best")
            flag = f" **{'/'.join(marks)}**" if marks else ""
            lines.append(
                f"| {point['threshold']}{flag} | "
                f"{point['correct'] + point['wrong'] + point['negatives']} | "
                f"{point['correct']} | {point['wrong']} | "
                f"{point['negatives']}/{point['negatives_total']} | "
                f"{point['precision']:.2f} | {point['recall']:.2f} | "
                f"{point['f1']:.2f} |"
            )
        lines.append("")
    lines += ["## Misses (right answer not ranked first)", ""]
    for mode, misses in report["misses"].items():
        if not misses:
            lines.append(f"- **{mode}**: none")
            continue
        listed = "; ".join(
            f"{m['entity']} ({m['kind']}) → {m['top']}, right answer at "
            f"{m['rank'] or 'not in top-k'}"
            for m in misses
        )
        lines.append(f"- **{mode}**: {listed}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--entities", type=Path, default=ENTITIES_PATH)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--backends",
        default="postgres,elasticsearch",
        help="Comma-separated: postgres, elasticsearch.",
    )
    parser.add_argument("--keep", action="store_true", help="Keep the collections.")
    parser.add_argument("--output", type=Path, help="Write the Markdown report here.")
    parser.add_argument("--json", type=Path, help="Write the report data here.")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    from auto_ontology.infra.elasticsearch import get_embed_endpoint
    from auto_ontology.utils.embedding import get_embed_params

    run = Run(
        backends=tuple(b.strip() for b in args.backends.split(",") if b.strip()),
        top_k=args.top_k,
        keep=args.keep,
    )
    run.embed_params = get_embed_params()
    if "elasticsearch" in run.backends:
        run.es_endpoint = get_embed_endpoint(f"auto-ontology-calibration-{run.suffix}")
    report = calibrate(run, load_entities(args.entities))
    markdown = render(report)
    if args.output:
        args.output.write_text(markdown)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
    print(markdown)


if __name__ == "__main__":
    main()
