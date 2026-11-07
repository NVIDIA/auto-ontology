# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end run of the Elasticsearch vector store on the local stack.

Starts Postgres and Elasticsearch with docker compose, seeds the pagila
fixture, and runs the ingestion service and the API on the host from this
checkout. Then drives them the way the UI does:

1. register pagila as a connection, which ingests it;
2. compile the semantic layer (the LLM names Terms and attributes).

and checks what was built:

* **Elasticsearch**: the license, the ``nvidia`` inference endpoint the app
  created, and how many rows each index holds.
* **Ontology**: the tables, columns, Terms and attributes in the catalog, with
  a sample of the Terms.
* **Retrieval**: the pagila data-layer entities from
  ``fixtures/calibration_entities.yaml`` searched through the app's own search
  function, scored by recall@1/3 and MRR; and the no-answer entities run
  through the relevance gate.
* **API**: global search and entity coverage over HTTP.

The cluster is the compose one, never an ``ES_URL`` from ``.env``: this script
overrides the vector-store and Postgres settings for its processes, and reads
only the model settings (``EMBED_*``, ``DEFAULT_MODELS_*``) from ``.env``.

Usage::

    uv run --no-sync python -m dev_tools.e2e_local --fresh --output e2e.md

``--fresh`` deletes the compose Postgres and Elasticsearch containers and
their volumes first. Without it the script refuses to run over an existing
pagila connection.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import httpx
from dotenv import dotenv_values

REPO_ROOT = Path(__file__).resolve().parent.parent
API_PORT = 3001
INGEST_PORT = 3002
DATABASE = "pagila"

#: Questions for the entity-coverage smoke test; the LLM extracts entities.
COVERAGE_QUESTIONS = (
    "Which customers rented the most films last month?",
    "What is the average rental rate per film category?",
    "How much revenue did each store make from payments?",
)
#: Global search is a text "contains" match in Postgres, not the vector store.
GLOBAL_SEARCH_TERMS = ("rental", "customer", "length")


def log(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


# ---------------------------------------------------------------------------
# Environment and infrastructure
# ---------------------------------------------------------------------------


def build_env(args: argparse.Namespace) -> dict[str, str]:
    """The environment every process here runs with.

    ``.env`` fills in what the shell does not set (the model settings, in
    practice); the vector store and Postgres are pinned to the compose
    services so a cluster configured in ``.env`` is never written to.
    """
    env = dict(os.environ)
    for key, value in dotenv_values(REPO_ROOT / ".env").items():
        env.setdefault(key, value or "")
    env.update(
        {
            "VDB_BACKEND": "elasticsearch",
            "VDB_SEARCH_MODE": args.search_mode,
            "ES_PORT": str(args.es_port),
            "ES_URL": f"http://localhost:{args.es_port}",
            "ES_API_KEY": "",
            "ES_USERNAME": "",
            "ES_PASSWORD": "",
            "ES_CA_CERTS": "",
            "ES_VERIFY_CERTS": "true",
            "ES_INDEX_PREFIX": "",
            "ES_INFERENCE_ID": "",
            "POSTGRES_HOST": "localhost",
            "POSTGRES_PORT": str(args.pg_port),
            "INGESTION_SERVICE_URL": f"http://127.0.0.1:{INGEST_PORT}",
            # Only the connection this script registers gets ingested.
            "CONNECTION_STRINGS": "",
        }
    )
    for key in ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DATABASE"):
        if not env.get(key):
            env[key] = "auto_ontology"
    # Compilation needs a chat model; an nvapi- embedding key reaches those too.
    if not (env.get("DEFAULT_MODELS_API_KEY") or env.get("NVIDIA_API_KEY")):
        env["DEFAULT_MODELS_API_KEY"] = env.get("EMBED_API_KEY", "")
    if not (env.get("EMBED_API_KEY") or env.get("DEFAULT_MODELS_API_KEY")):
        sys.exit("Set EMBED_API_KEY (an nvapi- key) in .env or the environment.")
    return env


def compose(env: dict[str, str], *args: str) -> None:
    subprocess.run(
        ["docker", "compose", "--profile", "elasticsearch", *args],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )


def reset_infra(env: dict[str, str]) -> None:
    """Remove the compose Postgres and Elasticsearch, volumes included."""
    log("--fresh: removing the compose postgres and elasticsearch and their data")
    compose(env, "rm", "-sfv", "postgres", "elasticsearch")
    config = subprocess.run(
        ["docker", "compose", "config", "--format", "json"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    project = json.loads(config.stdout)["name"]
    for volume in ("postgres_data", "elasticsearch_data"):
        subprocess.run(
            ["docker", "volume", "rm", "-f", f"{project}_{volume}"],
            capture_output=True,
            check=False,
        )


def start_infra(env: dict[str, str]) -> None:
    log("starting postgres and elasticsearch (docker compose)")
    compose(env, "up", "-d", "--wait", "postgres", "elasticsearch")
    py = [sys.executable, "-m"]
    log("seeding the pagila fixture")
    subprocess.run(
        [*py, "dev_tools.fixtures.seed_local_postgres"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )
    log("applying migrations")
    subprocess.run(
        [*py, "alembic", "upgrade", "head"], cwd=REPO_ROOT, env=env, check=True
    )


def enable_semantic_compilation() -> None:
    """Turn on the Settings toggle that semantic compilation runs behind.

    The flag lives in the frontend's ``configurations`` table, which Prisma
    creates and this script never runs. Without it every compilation pass is
    skipped ("compilation disabled in settings"). The table is created in the
    shape ``frontend/prisma/schema.prisma`` gives it, so a later ``prisma db
    push`` finds it already in place.
    """
    import psycopg

    from auto_ontology.infra.postgres import (
        FRONTEND_SCHEMA,
        get_postgres_connection_string,
    )
    from auto_ontology.ingestion_service.config import SEMANTIC_COMPILATION_ENABLED_KEY

    with psycopg.connect(get_postgres_connection_string()) as conn:
        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {FRONTEND_SCHEMA}")
        conn.execute(
            f"CREATE TABLE IF NOT EXISTS {FRONTEND_SCHEMA}.configurations ("
            " key text PRIMARY KEY, value text NOT NULL,"
            " updated_at timestamptz NOT NULL DEFAULT now())"
        )
        conn.execute(
            f"INSERT INTO {FRONTEND_SCHEMA}.configurations (key, value, updated_at)"
            " VALUES (%s, 'true', now())"
            " ON CONFLICT (key) DO UPDATE SET value = 'true', updated_at = now()",
            (SEMANTIC_COMPILATION_ENABLED_KEY,),
        )
    log("enabled semantic compilation (frontend.configurations)")


@contextlib.contextmanager
def services(
    env: dict[str, str], log_dir: Path, keep_running: bool
) -> Iterator[dict[str, Path]]:
    """Run the ingestion service and the API on the host; yield their log paths."""
    for port in (API_PORT, INGEST_PORT):
        with contextlib.suppress(httpx.HTTPError):
            httpx.get(f"http://127.0.0.1:{port}/", timeout=1)
            sys.exit(f"Port {port} is already in use; stop whatever is on it.")
    log_dir.mkdir(parents=True, exist_ok=True)
    logs = {"ingestion": log_dir / "ingestion.log", "api": log_dir / "api.log"}
    procs: list[subprocess.Popen] = []
    try:
        for name, module in (
            ("ingestion", "auto_ontology.ingestion_service"),
            ("api", "auto_ontology.server"),
        ):
            procs.append(
                subprocess.Popen(
                    [sys.executable, "-m", module],
                    cwd=REPO_ROOT,
                    env=env,
                    stdout=logs[name].open("w"),
                    stderr=subprocess.STDOUT,
                )
            )
        wait_for(f"http://127.0.0.1:{INGEST_PORT}/health", "ingestion service", procs)
        wait_for(f"http://127.0.0.1:{API_PORT}/api/health", "API", procs)
        yield logs
    finally:
        if keep_running:
            log(f"--keep-running: API on :{API_PORT}, ingestion on :{INGEST_PORT}")
        else:
            for proc in procs:
                proc.terminate()
            for proc in procs:
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=20)


def wait_for(url: str, name: str, procs: list[subprocess.Popen]) -> None:
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if any(p.poll() is not None for p in procs):
            sys.exit(f"A service exited while waiting for the {name}; see its log.")
        with contextlib.suppress(httpx.HTTPError):
            if httpx.get(url, timeout=3).status_code == 200:
                log(f"{name} is up")
                return
        time.sleep(2)
    sys.exit(f"The {name} did not become healthy at {url}.")


# ---------------------------------------------------------------------------
# Driving the app
# ---------------------------------------------------------------------------


def register_connection(env: dict[str, str], reuse: bool) -> bool:
    """Register pagila; False when it already was and *reuse* allows that."""
    connection = {
        "type": "postgresql",
        "host": env["POSTGRES_HOST"],
        "port": env["POSTGRES_PORT"],
        "user": env["POSTGRES_USER"],
        "password": env["POSTGRES_PASSWORD"],
        "database": DATABASE,
    }
    response = httpx.post(
        f"http://127.0.0.1:{API_PORT}/api/connections",
        json={"connection": connection},
        timeout=60,
    )
    if response.status_code == 422 and "already exists" in response.text:
        if reuse:
            log(f"{DATABASE} is already connected; --reuse: skipping ingest")
            return False
        sys.exit(f"{DATABASE} is already connected; rerun with --reuse or --fresh.")
    response.raise_for_status()
    log(f"registered the {DATABASE} connection; ingest started")
    return True


def wait_for_log(
    path: Path, done: str, failed: str, timeout_s: float, what: str
) -> float:
    """Seconds until *done* appears in the log at *path*; exits on *failed*."""
    started = time.monotonic()
    deadline = started + timeout_s
    while time.monotonic() < deadline:
        text = path.read_text(errors="replace")
        if re.search(failed, text):
            sys.exit(f"{what} failed; see {path}")
        if re.search(done, text):
            return time.monotonic() - started
        time.sleep(3)
    sys.exit(f"{what} did not finish within {timeout_s:.0f}s; see {path}")


def compile_semantic_layer(timeout_s: float) -> float:
    base = f"http://127.0.0.1:{INGEST_PORT}"
    started_at = datetime.now().astimezone()
    started = time.monotonic()
    if httpx.get(f"{base}/semantic/status", timeout=30).json().get("running"):
        log("semantic compilation already running; waiting for it")
    else:
        httpx.post(f"{base}/semantic/compile", timeout=30).raise_for_status()
        log("semantic compilation started")
    last_note = started
    while time.monotonic() - started < timeout_s:
        time.sleep(5)
        status = httpx.get(f"{base}/semantic/status", timeout=30).json()
        failure = _parse_time(status.get("last_failure_at"))
        success = _parse_time(status.get("last_success_at"))
        if failure and failure >= started_at:
            sys.exit("Semantic compilation failed; see the ingestion log.")
        if not status.get("running") and success and success >= started_at:
            return time.monotonic() - started
        if time.monotonic() - last_note > 60:
            log(f"  still compiling ({time.monotonic() - started:.0f}s)")
            last_note = time.monotonic()
    sys.exit(f"Semantic compilation did not finish within {timeout_s:.0f}s.")


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.astimezone()


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_elasticsearch() -> dict[str, Any]:
    from auto_ontology.infra.elasticsearch import (
        get_elasticsearch_client,
        get_elasticsearch_config,
    )
    from auto_ontology.vdb import DATA_VDB_COLLECTION, SEMANTIC_VDB_COLLECTION

    config = get_elasticsearch_config()
    client = get_elasticsearch_client(config)
    client.indices.refresh(index=f"{config.index_prefix}-*")
    endpoint = client.inference.get(
        task_type="text_embedding", inference_id=config.inference_id
    )["endpoints"][0]
    counts = {
        collection: client.count(index=f"{config.index_prefix}-{collection}")["count"]
        for collection in (DATA_VDB_COLLECTION, SEMANTIC_VDB_COLLECTION)
    }
    return {
        "version": client.info()["version"]["number"],
        "license": client.license.get()["license"]["type"],
        "inference_id": endpoint["inference_id"],
        "service": endpoint["service"],
        "model_id": endpoint["service_settings"].get("model_id"),
        "similarity": endpoint["service_settings"].get("similarity"),
        "dimensions": endpoint["service_settings"].get("dimensions"),
        "counts": counts,
    }


def check_ontology() -> dict[str, Any]:
    from auto_ontology.dal.datasources import fetch_schemas_by_ids
    from auto_ontology.dal.terms import fetch_all_terms_and_attributes

    columns = [r for r in fetch_schemas_by_ids() if r["database_name"] == DATABASE]
    terms, attributes = fetch_all_terms_and_attributes()
    return {
        "tables": len({(r["table_schema"], r["table_name"]) for r in columns}),
        "columns": len(columns),
        "terms": len(terms),
        "attributes": len(attributes),
        "term_sample": [
            {
                "name": t.get("name"),
                "description": (t.get("description") or "")[:120],
            }
            for t in sorted(terms, key=lambda t: str(t.get("name")))[:25]
        ],
    }


def check_retrieval(max_distance: float) -> dict[str, Any]:
    """Pagila data-layer entities, searched the way the app searches."""
    from dev_tools.calibrate_vdb import (
        Run,
        first_correct_rank,
        load_entities,
        ranking_metrics,
        resolve_targets,
        search,
    )
    from auto_ontology.retrieval.data_access.hit_scores import (
        cosine_distance,
        passes_relevance_gate,
    )
    from auto_ontology.utils import get_embed_params
    from auto_ontology.vdb import get_data_vdb

    entities = [
        e for e in load_entities() if e.layer == "data" and e.database == DATABASE
    ]
    targets = resolve_targets(entities)
    store = get_data_vdb()
    run = Run(embed_params=get_embed_params())
    ranks: list[int | None] = []
    misses: list[dict[str, Any]] = []
    false_accepts: list[dict[str, Any]] = []
    no_answer = 0
    for entity in entities:
        rows = search(store, run, entity)
        top = rows[0] if rows else None
        if entity.has_answer:
            rank = first_correct_rank(rows, targets[entity.entity])
            ranks.append(rank)
            if rank != 1:
                misses.append(
                    {
                        "entity": entity.entity,
                        "rank": rank,
                        "top": _hit_name(top),
                        "expected": list(entity.expect),
                    }
                )
        else:
            no_answer += 1
            if top and passes_relevance_gate(top, max_distance):
                false_accepts.append(
                    {
                        "entity": entity.entity,
                        "top": _hit_name(top),
                        "distance": round(cosine_distance(top), 3),
                    }
                )
    return {
        "metrics": ranking_metrics(ranks, run.top_k),
        "misses": misses,
        "no_answer": no_answer,
        "false_accepts": false_accepts,
        "max_distance": max_distance,
    }


def _hit_name(hit: dict[str, Any] | None) -> str | None:
    if not hit:
        return None
    parts = [hit.get("schema_name"), hit.get("table_name"), hit.get("name")]
    return ".".join(str(p) for p in parts if p)


def check_api() -> dict[str, Any]:
    base = f"http://127.0.0.1:{API_PORT}/api"
    searches = {}
    for term in GLOBAL_SEARCH_TERMS:
        response = httpx.post(
            f"{base}/search/global-search/count", json={"search_term": term}, timeout=60
        )
        searches[term] = (
            response.json() if response.is_success else response.status_code
        )
    coverage = []
    for question in COVERAGE_QUESTIONS:
        response = httpx.post(
            f"{base}/question-entity-coverage", json={"question": question}, timeout=300
        )
        coverage.append(
            {
                "question": question,
                "status": response.status_code,
                "body": response.json() if response.is_success else response.text[:300],
            }
        )
    return {"global_search_counts": searches, "coverage": coverage}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def render(report: dict[str, Any]) -> str:
    es, onto, ret = report["elasticsearch"], report["ontology"], report.get("retrieval")
    lines = [
        f"# Local end-to-end run ({report['search_mode']} search)",
        "",
        (
            f"Ingest took {report['ingest_s']:.0f}s"
            if report.get("ingest_s") is not None
            else "Ingest reused from an earlier run"
        )
        + (
            f", semantic compilation {report['compile_s']:.0f}s."
            if report.get("compile_s") is not None
            else "; semantic compilation skipped."
        ),
        "",
        "## Elasticsearch",
        "",
        f"- {es['version']}, {es['license']} license",
        f"- Endpoint `{es['inference_id']}`: {es['service']} / `{es['model_id']}`, "
        f"{es['dimensions']} dims, {es['similarity']}",
        *(f"- `{name}`: {count} rows" for name, count in es["counts"].items()),
        "",
        "## Ontology",
        "",
        f"- {onto['tables']} tables and views, {onto['columns']} columns in {DATABASE}",
        f"- {onto['terms']} Terms, {onto['attributes']} attributes",
        "",
        "| Term | Description |",
        "|---|---|",
        *(
            f"| {t['name']} | {t['description'].replace('|', '/')} |"
            for t in onto["term_sample"]
        ),
    ]
    if ret:
        m = ret["metrics"]
        lines += [
            "",
            "## Retrieval (pagila data layer)",
            "",
            f"- {m['n']} entities with an answer: recall@1 {m['recall@1']:.2f}, "
            f"recall@3 {m['recall@3']:.2f}, MRR {m['mrr']:.2f}",
            f"- {ret['no_answer']} with no answer: {len(ret['false_accepts'])} "
            f"accepted at max_distance {ret['max_distance']}",
        ]
        if ret["misses"]:
            lines += ["", "Not ranked first:", ""]
            lines += [
                f"- {x['entity']!r}: rank {x['rank']}, top `{x['top']}`, "
                f"expected {', '.join(x['expected'])}"
                for x in ret["misses"]
            ]
        if ret["false_accepts"]:
            lines += ["", "No-answer entities accepted:", ""]
            lines += [
                f"- {x['entity']!r} -> `{x['top']}` at {x['distance']}"
                for x in ret["false_accepts"]
            ]
    api = report.get("api")
    if api:
        lines += ["", "## API", "", "Global search counts:", ""]
        lines += [f"- {term!r}: {n}" for term, n in api["global_search_counts"].items()]
        lines += ["", "Entity coverage:", ""]
        for item in api["coverage"]:
            body = json.dumps(item["body"])[:600]
            lines += [f"- {item['question']} ({item['status']}): `{body}`"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--fresh", action="store_true", help="Wipe compose PG/ES first."
    )
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Keep an existing pagila connection and its ingest; compile and check.",
    )
    parser.add_argument(
        "--search-mode", choices=("semantic", "hybrid"), default="semantic"
    )
    parser.add_argument("--es-port", type=int, default=9201)
    parser.add_argument("--pg-port", type=int, default=5432)
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--ingest-timeout", type=float, default=900)
    parser.add_argument("--compile-timeout", type=float, default=3600)
    parser.add_argument("--max-distance", type=float, default=0.65)
    parser.add_argument("--keep-running", action="store_true")
    parser.add_argument("--log-dir", type=Path, default=REPO_ROOT / ".e2e")
    parser.add_argument("--output", type=Path, help="Write the Markdown report here.")
    args = parser.parse_args()

    env = build_env(args)
    # The checks below import the app in this process; it must see the same
    # settings as the services (auto_ontology.env never overrides os.environ).
    os.environ.update(env)

    if args.fresh:
        reset_infra(env)
    start_infra(env)

    report: dict[str, Any] = {"search_mode": args.search_mode}
    with services(env, args.log_dir, args.keep_running) as logs:
        if register_connection(env, args.reuse):
            report["ingest_s"] = wait_for_log(
                logs["ingestion"],
                rf"Data ingestion finished successfully for database {DATABASE}\b",
                rf"(Background ingest failed|Ingest skipped).*{DATABASE}",
                args.ingest_timeout,
                "Ingest",
            )
            log(f"ingest finished in {report['ingest_s']:.0f}s")
        if not args.skip_compile:
            enable_semantic_compilation()
            report["compile_s"] = compile_semantic_layer(args.compile_timeout)
            log(f"semantic compilation finished in {report['compile_s']:.0f}s")
        report["elasticsearch"] = check_elasticsearch()
        report["ontology"] = check_ontology()
        report["retrieval"] = check_retrieval(args.max_distance)
        report["api"] = check_api()

    text = render(report)
    if args.output:
        args.output.write_text(text)
        log(f"report written to {args.output}")
    print(text)


if __name__ == "__main__":
    main()
