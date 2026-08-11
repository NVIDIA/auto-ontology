# GSF (Generative Semantic Fabric)

Full-stack repo: Next.js frontend (`/frontend`) + FastAPI backend (`/gsf`).

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

- **In-flight refactor — dropping Neo4j.** Read `docs/refactor/drop-neo4j/PLAN.md` before changing anything under `gsf/dal/` or `gsf/catalog/`. Log what you land in `PROGRESS.md` **in the same commit**, and record any deviation from the plan in `DECISIONS.md` *before* the code lands. Do not silently absorb behaviour changes — surface them.
- **Formatting/Linting**: Ruff (line-length: 88). Run `uv run ruff check gsf/` and `uv run ruff format gsf/` from repo root before committing.
- **Dependencies**: Managed with `uv`. Add dependencies via `uv add`, not pip. Do not edit `pyproject.toml` manually for deps.
- **API prefix**: All routes under `/api/`. Routes live in `gsf/server/datasources/router.py`, orchestration in `gsf/server/datasources/service.py`.
- **Type hints**: Required on all function signatures.
- **Exception handlers**: Use plain `def` (not `async def`) for `@app.exception_handler` functions. They perform no async I/O, so synchronous handlers are preferred.

## Git

- Commit messages: concise imperative style, no `Co-Authored-By` trailers.
- Branch from `main`. Current working branch: `fix/architecture`.
