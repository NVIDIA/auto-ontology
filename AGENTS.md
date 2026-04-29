# GSF (Generative Semantic Fabric)

Full-stack repo: Next.js frontend (`/frontend`) + FastAPI backend (`/gsf`).

## Structure

```
/frontend/    # Next.js 16 app (React 19, TypeScript, Tailwind CSS 4)
/gsf/         # FastAPI app (Python 3.11+, uv)
```

## Dev Commands

| Command | Description |
|---|---|
| `pnpm dev` | Start Next.js on :3000 |
| `pnpm dev:api` | Start FastAPI on :3001 |
| `pnpm build` | Build the frontend |
| `pnpm lint` | ESLint the frontend |

To install dependencies:
- Frontend: `pnpm install` from repo root
- Backend: `uv sync` from repo root

## Frontend Rules

- **Next.js 16**: Has breaking changes from prior versions. Read `node_modules/next/dist/docs/` before writing Next.js-specific code. See `frontend/AGENTS.md` for details.
- **Formatting**: Prettier — tabs, single quotes, semicolons, 100 char width, trailing commas. Run `prettier --write` before committing.
- **Linting**: ESLint (Airbnb config). Fix all lint errors; do not disable rules without a comment explaining why.
- **Components**: Arrow function components. No class components.
- **Styling**: Tailwind CSS 4. Do not write inline styles or separate CSS files for component styling.
- **Types**: TypeScript strict mode. No `any` without a comment. Types go in `/frontend/types`, enums in `/frontend/enums`.
- **Testing**: Vitest + Playwright.
- **API layer**: Use the `requests` wrapper from `frontend/api/requests.ts` — do not call axios directly. Base URL and error handling are already centralised there.

## Backend Rules

- **Formatting/Linting**: Ruff (line-length: 88). Run `uv run ruff check gsf/` and `uv run ruff format gsf/` from repo root before committing.
- **Dependencies**: Managed with `uv`. Add dependencies via `uv add`, not pip. Do not edit `pyproject.toml` manually for deps.
- **API prefix**: All routes under `/api/`. Routes live in `gsf/server/datasources/router.py`, data access in `gsf/server/datasources/dal.py`.
- **Type hints**: Required on all function signatures.
- **Exception handlers**: Use plain `def` (not `async def`) for `@app.exception_handler` functions. They perform no async I/O, so synchronous handlers are preferred.

## Git

- Commit messages: concise imperative style, no `Co-Authored-By` trailers.
- Branch from `main`. Current working branch: `fix/architecture`.

## NVIDIA Agent Skills

Vendored skills from the [NVIDIA skills catalog](https://github.com/NVIDIA/skills) live under `.cursor/skills/nvidia/`. Read and follow them when the relevant task comes up:

| Skill | Path | Use when |
|---|---|---|
| RAG Blueprint | `.cursor/skills/nvidia/RAG-Blueprint/rag-blueprint/SKILL.md` | Deploying, configuring, troubleshooting, or tearing down the RAG pipeline (ingestion, retrieval, guardrails, query rewriting, observability). Docker Compose or Helm. |
| NeMo Evaluator — BYOB | `.cursor/skills/nvidia/NeMo-Evaluator/byob/SKILL.md` | Adding a bring-your-own benchmark for evaluating chat/RAG quality. |
| NeMo Evaluator Launcher — launching-evals | `.cursor/skills/nvidia/NeMo-Evaluator-Launcher/launching-evals/SKILL.md` | Kicking off LLM evaluation runs via the launcher. |
| NeMo Evaluator Launcher — accessing-mlflow | `.cursor/skills/nvidia/NeMo-Evaluator-Launcher/accessing-mlflow/SKILL.md` | Reading/compare eval results stored in MLflow. |
| NeMo Evaluator Launcher — nel-assistant | `.cursor/skills/nvidia/NeMo-Evaluator-Launcher/nel-assistant/SKILL.md` | End-to-end assistant for the NeMo Evaluator Launcher workflow. |

These are static copies. To refresh, re-pull from `github.com/NVIDIA/skills` (NeMo-Evaluator*) and `github.com/NVIDIA-AI-Blueprints/rag` (RAG Blueprint, from `skill-source/.agents/skills/rag-blueprint`).
