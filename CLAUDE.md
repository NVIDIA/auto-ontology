<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Auto Ontology

Full-stack repo: Next.js frontend (`/frontend`) + FastAPI backend (`/auto_ontology`).

This file is for **Auto Ontology maintainers**. User/partner agent skills live in
`skills/` — start at [`AGENTS.md`](./AGENTS.md).

## Dev Commands

Backend commands run from the repo root:

| Command | Description |
|---|---|
| `uv run uvicorn auto_ontology.server.__main__:create_app --factory --reload --host 127.0.0.1 --port 3001` | Start FastAPI on :3001 with hot reload |
| `uv run python -m auto_ontology.server` | Start FastAPI on :3001 without reload (what the container runs) |
| `uv run python -m auto_ontology.ingestion_service` | Start the ingestion service on :3002 |

The app factory is `create_app()` in `auto_ontology/server/__main__.py` — there is no
`auto_ontology/server/main.py`.

## Frontend Rules

In [`frontend/CLAUDE.md`](./frontend/CLAUDE.md), loaded when working under `frontend/`.

## Backend Rules

- **The store is Postgres.** `auto_ontology/dal/` queries it with SQLAlchemy Core (no ORM); `auto_ontology/dal/schema.py` is the single source of truth for the schema and `auto_ontology/dal/session.py` owns the pooled engine. Schema changes go through Alembic — `uv run alembic revision --autogenerate -m "..."` — never by hand-editing a migration that has been applied. Several column types and constraints are deliberate and non-obvious — read the comments in `schema.py` before changing a table.
- **Formatting/Linting**: Run `uv run ruff check auto_ontology/` and `uv run ruff format auto_ontology/` from repo root before committing.
- **Dependencies**: Managed with `uv`. Add dependencies via `uv add`, not pip. Do not edit `pyproject.toml` manually for deps.
- **API prefix**: All routes under `/api/`. Routes live in `auto_ontology/server/datasources/router.py`, orchestration in `auto_ontology/server/datasources/service.py`.
- **Type hints**: Required on all function signatures.
- **Exception handlers**: Use plain `def` (not `async def`) for `@app.exception_handler` functions. They perform no async I/O, so synchronous handlers are preferred.

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
|---|---|---|
| Ruff | `uv run ruff check auto_ontology/ && uv run ruff format --check auto_ontology/` | Backend lint/format |
| Pytest | `uv run pytest auto_ontology dev_tools -q` | Backend tests |
| OpenAPI | `uv run python -m dev_tools.generate_backend_openapi` | **Stale `docs/openapi/*.json`** |
| Frontend | `cd frontend && npx prettier --check . && pnpm lint && pnpm build` | Format, lint, build |

The OpenAPI one is the easy one to miss: `docs/openapi/backend.json` and
`ingestion.json` are committed, and the spec embeds each route's **docstring**.
Editing a docstring on any `@router` handler — not just changing a signature or
model — makes the committed spec stale and fails "Backend spec is up to date".
Regenerate and commit the result whenever a route's code or docs change.
