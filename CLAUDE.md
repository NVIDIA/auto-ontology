<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Auto-Ontology

Full-stack repo: Next.js frontend (`/frontend`) + FastAPI backend (`/auto_ontology`),
plus a separate read-only MCP server (`/mcp`).

This file is for **Auto-Ontology maintainers**. User/partner agent skills live in
`skills/` — start at [`AGENTS.md`](./AGENTS.md).

## Architecture

Four services; the first three share one Postgres (+pgvector) instance:

| Service | Tech | Port | Role |
| --- | --- | --- | --- |
| Frontend | Next.js 16, Better Auth, Prisma | 3000 | Web UI **and the public API gateway** |
| Backend API | FastAPI (Python 3.12+), LangChain | 3001 | Chat / NL-to-SQL, catalog, ontology APIs |
| Ingestion service | FastAPI (Python 3.12+) | 3002 | Ingestion + semantic compilation; no auth of its own |
| MCP server | FastMCP (`mcp/` — its own uv project, venv, and CI workflow) | 3003 | Read-only tools over the public API |

**Trust boundary (non-negotiable).** The Python services are private/ClusterIP-only.
FastAPI trusts the `x-auto-ontology-user-id` header forwarded by the Next.js gateway
(`auto_ontology/server/identity.py`) and does not re-authenticate it; the ingestion
endpoints have no authentication at all. Never expose :3001/:3002 — all client traffic
goes through :3000. Request flow: browser → `frontend/app/api/**/route.ts`
(permission via `withPermission`) → FastAPI.

**Postgres ownership is split.** The backend owns the `auto_ontology` schema
(migrated by Alembic); the frontend's Prisma owns `public` (`frontend/prisma`,
migrated separately). The two are independent — don't cross them.

Backend layout (big picture):

- `auto_ontology/dal/` — all Postgres access, SQLAlchemy Core (no ORM). `schema.py`
  is the single source of truth; `session.py` owns the pooled engine.
- `auto_ontology/server/<area>/router.py` — one FastAPI router per area, all mounted
  under `/api` in `server/__main__.py`. The app factory is `create_app()` there;
  there is no `auto_ontology/server/main.py`.
- `auto_ontology/semantic/` — the semantic-compilation pipeline: term extraction,
  relationship/SQL-attribute suggestion, embeddings.
- `auto_ontology/connectors/` — the ten supported engines behind a registry
  (`connection_string_factory.py`); credentials optionally via Vault (`vault.py`).
- `auto_ontology/retrieval/` + `auto_ontology/vdb/` — pgvector-backed candidate
  retrieval for NL-to-SQL; `retrieval/entity_coverage/` grades whether the
  ontology can answer a question at all, `retrieval/kumo/` routes predictive
  questions to KumoRFM.
- `auto_ontology/catalog/` — the data catalog: extracts schemas, column profiles,
  and query history from sources through a connector and writes them (SQL and
  storage types are confined to `store/`; everything above it is storage-agnostic).
- `auto_ontology/pii_detection/` — rules-first PII classification over catalog
  columns, with an LLM classifier backend.
- `auto_ontology/server/model_interchange/` — native and Apache Ossie model YAML
  import/export (`/api/model/import`, `/api/model/export`) via the
  `apache-ossie-nvidia-auto-ontology` git dependency.
- `auto_ontology/ingestion_service/` — the :3002 app (`python -m auto_ontology.ingestion_service`).

The ontology itself stores three kinds of definition — terms, SQL attributes
(reviewed SQL expressions), and relationships — with pgvector embeddings for
retrieval. `README.md` ("How it works") has the domain story; `AGENTS.md` maps
user workflows to the `skills/` directories. Feature docs — authentication,
connections, the conversation API, MCP — live in `docs/`.

## Dev Commands

Backend commands run from the repo root:

| Command | Description |
| --- | --- |
| `./dev_tools/setup_env.sh` | Full stack via docker compose (add `--dev` for infra only, `--ds` for docker frontend + local backend) |
| `uv run uvicorn auto_ontology.server.__main__:create_app --factory --reload --host 127.0.0.1 --port 3001` | Start FastAPI on :3001 with hot reload |
| `uv run python -m auto_ontology.server` | Start FastAPI on :3001 without reload (what the container runs) |
| `uv run python -m auto_ontology.ingestion_service` | Start the ingestion service on :3002 |
| `uv run pytest auto_ontology dev_tools -q` | Full backend suite |
| `uv run pytest auto_ontology/tests/test_literal_check.py::test_name` | Single test |
| `make migrate` / `make migrate-check` / `make migrate-revision m="..."` | Alembic upgrade / drift check / autogenerate revision |

Tests that hit Postgres (`auto_ontology/dal/tests/test_golden.py`,
`auto_ontology/connectors/tests/test_postgres.py`) replay fixture databases seeded
from `dev_tools/fixtures/` (`seed_fixtures`, then `alembic upgrade head`, then
`seed_graph_fixture --reset`) — CI depends on those fixtures. Seed recipe and what
each database covers: [`dev_tools/README.md`](./dev_tools/README.md).

CI syncs with `uv sync --locked --extra kumo`; the `kumo` extra (wheels vendored
in `vendor/wheels`) is what runs the KumoRFM tests — locally they skip silently
without it.

The root pytest suite deliberately ignores `mcp/` (`--ignore=mcp` in `pyproject.toml`):
it is a separate project with its own venv and CI workflow, and letting it onto
`pythonpath` would shadow the installed `mcp` SDK. Run its tests from within `mcp/`.

## Frontend Rules

In [`frontend/CLAUDE.md`](./frontend/CLAUDE.md), loaded when working under `frontend/`.
Note Next.js 16 has breaking changes vs. training data — see the notice in
`frontend/AGENTS.md`.

## Backend Rules

- **The store is Postgres.** `auto_ontology/dal/` queries it with SQLAlchemy Core (no ORM). Schema changes go through Alembic (`make migrate-revision m="..."`; migrations live in `alembic/versions/` at the repo root), never by hand-editing a migration that has been applied. Several column types and constraints are deliberate and non-obvious — read the comments in `schema.py` before changing a table. `make migrate-check` (CI) fails when `schema.py` and the database disagree — a schema change committed without its migration is invisible until deploy.
- **Formatting/Linting**: Run `uv run ruff check auto_ontology/ alembic/ dev_tools/` and `uv run ruff format auto_ontology/ alembic/ dev_tools/` from repo root before committing — CI lints all three directories, not just `auto_ontology/`. The ruff rule set is pinned to `E4,E7,E9,F` on purpose (see the comment in `pyproject.toml`) — don't widen it casually or bulk-`--fix` newer rule families.
- **Dependencies**: Managed with `uv`. Add dependencies via `uv add`, not pip. Do not edit `pyproject.toml` manually for deps.
- **API prefix**: All routes under `/api/`. Routes live in `auto_ontology/server/<area>/router.py`, orchestration in a sibling `service.py` where one exists.
- **Type hints**: Required on all function signatures.
- **Exception handlers**: Use plain `def` (not `async def`) for `@app.exception_handler` functions. They perform no async I/O, so synchronous handlers are preferred.
- **SPDX headers**: Every new file carries the NVIDIA copyright + Apache-2.0 SPDX header used across the repository — copy it from any existing file.
- **Models**: LLM/embedding/rerank endpoints are configured via env triplets (`<ROLE>_ENDPOINT`/`<ROLE>_API_KEY`/`<ROLE>_MODEL` for `REASONING`, `NON_REASONING`, `EMBED`, `RERANK`), each falling back to `DEFAULT_MODELS_*`. `.env.example` documents every variable.

## Git

- Commit messages: concise imperative style, no `Co-Authored-By` trailers.
- Branch from `main`.

### Opening a PR

**A PR is not done until every GitHub Actions workflow passes.** After opening
one, poll `gh pr checks <number>` until no check is pending, then fix whatever
failed and push again — repeat until the run is green. Report the final state
honestly; do not hand back a PR with unexamined or failing checks.

Run these locally first — they are the same gates CI applies, and catching a
failure here costs seconds instead of a CI round-trip:

| Check | Command | Catches |
| --- | --- | --- |
| Ruff | `uv run ruff check auto_ontology/ alembic/ dev_tools/ && uv run ruff format --check auto_ontology/ alembic/ dev_tools/` | Backend lint/format |
| Pytest | `uv run pytest auto_ontology dev_tools -q` | Backend tests |
| OpenAPI (backend) | `uv run python -m dev_tools.generate_backend_openapi` | **Stale `docs/openapi/backend.json` / `ingestion.json`** |
| OpenAPI (public) | `cd frontend && pnpm openapi` | **Stale `docs/openapi/auto-ontology-api.json`** |
| Frontend | `cd frontend && npx prettier --check . && pnpm lint && pnpm build` | Format, lint, build |

The OpenAPI ones are the easy ones to miss. `docs/openapi/*.json` are committed
generated files — never edit by hand:

- `backend.json` + `ingestion.json` describe the **internal** FastAPI apps and
  embed each route's **docstring**. Editing a docstring on any `@router` handler —
  not just changing a signature or model — makes the committed spec stale.
  Regenerate with `dev_tools.generate_backend_openapi`.
- `auto-ontology-api.json` is the **public** spec, generated from
  `frontend/app/api/**/route.ts` by `pnpm openapi`. Regenerate it whenever you add
  or change a frontend API route or its `withPermission` wrapper — it also writes
  the packaged copy `mcp/auto_ontology_mcp/auto-ontology-api.json`, and CI diffs both.
