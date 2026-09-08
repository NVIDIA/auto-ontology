# GSF (Generative Semantic Fabric)

Full-stack repo: Next.js frontend (`/frontend`) + FastAPI backend (`/gsf`).

This file is for **GSF maintainers**. User/partner agent skills live in
`skills/` — start at [`AGENTS.md`](./AGENTS.md).

## Structure

```
/frontend/ # Next.js 16 app (React 19, TypeScript, Tailwind CSS 4)
/gsf/      # FastAPI app (Python 3.11+, uv)
```

## Dev Commands

Frontend commands run from `/frontend`:

| Command | Description |
|---|---|
| `pnpm dev` | Start Next.js on :3000 |
| `pnpm build` | Build the frontend |
| `pnpm lint` | ESLint the frontend |

Backend commands run from the repo root:

| Command | Description |
|---|---|
| `uv run uvicorn gsf.server.__main__:create_app --factory --reload --host 127.0.0.1 --port 3001` | Start FastAPI on :3001 with hot reload |
| `uv run python -m gsf.server` | Start FastAPI on :3001 without reload (what the container runs) |
| `uv run python -m gsf.ingestion_service` | Start the ingestion service on :3002 |

The app factory is `create_app()` in `gsf/server/__main__.py` — there is no
`gsf/server/main.py`.

To install dependencies:
- Frontend: `pnpm install` from `/frontend`
- Backend: `uv sync` from repo root

## Frontend Rules

- **Next.js 16**: Has breaking changes from prior versions. Read `node_modules/next/dist/docs/` before writing Next.js-specific code. See `frontend/AGENTS.md` for details.
- **Formatting**: Prettier — tabs, single quotes, semicolons, 100 char width, trailing commas. **After editing ANY frontend file, ALWAYS run `pnpm format` (or `cd frontend && npx prettier --write <file>`) to auto-format it.** Never rely on manual formatting — always let Prettier handle it. CI will reject unformatted code.
- **Linting**: ESLint (Airbnb config). Fix all lint errors; do not disable rules without a comment explaining why.
- **Components**: Arrow function components. No class components.
- **Styling**: Tailwind CSS 4. Do not write inline styles or separate CSS files for component styling.
- **Types**: TypeScript strict mode. No `any` without a comment. Types go in `/frontend/types`, enums in `/frontend/enums`.
- **Testing**: Vitest + Playwright.
- **API layer**: Use the `requests` wrapper from `frontend/api/requests.ts` — do not call axios directly. Base URL and error handling are already centralised there.

## Backend Rules

- **The store is Postgres.** `gsf/dal/` queries it with SQLAlchemy Core (no ORM); `gsf/dal/schema.py` is the single source of truth for the schema and `gsf/dal/session.py` owns the pooled engine. Schema changes go through Alembic — `uv run alembic revision --autogenerate -m "..."` — never by hand-editing a migration that has been applied. Several column types and constraints are deliberate and non-obvious — read the comments in `schema.py` before changing a table.
- **Formatting/Linting**: Ruff (line-length: 88). Run `uv run ruff check gsf/` and `uv run ruff format gsf/` from repo root before committing.
- **Dependencies**: Managed with `uv`. Add dependencies via `uv add`, not pip. Do not edit `pyproject.toml` manually for deps.
- **API prefix**: All routes under `/api/`. Routes live in `gsf/server/datasources/router.py`, orchestration in `gsf/server/datasources/service.py`.
- **Type hints**: Required on all function signatures.
- **Exception handlers**: Use plain `def` (not `async def`) for `@app.exception_handler` functions. They perform no async I/O, so synchronous handlers are preferred.

## Git

- Commit messages: concise imperative style, no `Co-Authored-By` trailers.
- Branch from `main`. Current working branch: `fix/architecture`.

### Opening a PR

**A PR is not done until every GitHub Actions workflow passes.** After opening
one, poll `gh pr checks <number>` until no check is pending, then fix whatever
failed and push again — repeat until the run is green. Report the final state
honestly; do not hand back a PR with unexamined or failing checks.

Run these locally first — they are the same gates CI applies, and catching a
failure here costs seconds instead of a CI round-trip:

| Check | Command | Catches |
|---|---|---|
| Ruff | `uv run ruff check gsf/ && uv run ruff format --check gsf/` | Backend lint/format |
| Pytest | `uv run pytest gsf dev_tools -q` | Backend tests |
| OpenAPI | `uv run python -m dev_tools.generate_backend_openapi` | **Stale `docs/openapi/*.json`** |
| Frontend | `cd frontend && npx prettier --check . && pnpm lint && pnpm build` | Format, lint, build |

The OpenAPI one is the easy one to miss: `docs/openapi/backend.json` and
`ingestion.json` are committed, and the spec embeds each route's **docstring**.
Editing a docstring on any `@router` handler — not just changing a signature or
model — makes the committed spec stale and fails "Backend spec is up to date".
Regenerate and commit the result whenever a route's code or docs change.
