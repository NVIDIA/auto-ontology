# dev_tools

Local development helpers: environment setup, database seeding, and ingest.

## Layout

| Path | What it is |
|---|---|
| `setup_env.sh` | Brings the docker-compose stack up. Human-invoked. |
| `local_ingest.py` | Ingests the local Postgres source into the pgvector store. |
| `generate_backend_openapi.py` | Dumps the FastAPI OpenAPI specs to `docs/openapi/`. Run by `ci-openapi.yml`. |
| `release_helm_chart.py` | Packages the chart and publishes it to the NGC Helm registry. Run by `release-helm.yml`. |
| `_add_spdx_headers.py` | One-off OSRB helper; kept so the header insertion is reproducible. |
| `fixtures/` | The test fixtures. **CI depends on these** — see below. |

Everything under `fixtures/` builds the databases that `auto_ontology/dal/tests/test_golden.py`
and `auto_ontology/connectors/tests/test_postgres.py` replay against, and is driven by
`.github/workflows/ci-python.yml`. Changing it changes what CI proves. See
[`fixtures/sql/README.md`](./fixtures/sql/README.md) for what each fixture
database covers and why.

```bash
docker compose up -d postgres
uv run --no-sync python -m dev_tools.fixtures.seed_fixtures          # pagila + chinook
uv run --no-sync alembic upgrade head
uv run --no-sync python -m dev_tools.fixtures.seed_graph_fixture --reset
uv run --no-sync pytest auto_ontology dev_tools -q
```

## Running locally

The **frontend** (`frontend/`) and the **Python API** (`auto_ontology/`) must both be running.

### One-time setup

```bash
cp .env.example .env   # then edit credentials as needed
cd frontend && pnpm install
uv sync                # from the repo root
```

### Start development

Full stack (infrastructure + Next.js + FastAPI):

```bash
./dev_tools/setup_env.sh
```

Or infrastructure only (Postgres, pgAdmin, ingestion service):

```bash
./dev_tools/setup_env.sh --dev
```

Then start the app manually:

```bash
cd frontend && pnpm dev   # Next.js on port 3000
uv run python -m auto_ontology.server   # FastAPI on port 3001 (from the repo root)
```

Open **http://localhost:3000**.

### Optional

- Override the API URL: set **`PYTHON_API_URL`** (used by Next rewrites and server-side API calls).

## `setup_env.sh`

Brings up the Auto Ontology environment via `docker compose`.

| Command | What it starts |
|---|---|
| `bash ./dev_tools/setup_env.sh` | Full stack — infra (Postgres, pgAdmin, ingestion service) **and** the `auto_ontology` + `auto-ontology-frontend` images (built with `--build`). |
| `bash ./dev_tools/setup_env.sh --dev` | Infra only. You run `auto_ontology` / `auto-ontology-frontend` locally yourself. |
| `bash ./dev_tools/setup_env.sh --ds` | Infra **+** `auto-ontology-frontend` in docker. You run `auto_ontology` locally on `:3001`; the containerised frontend reaches it via `host.docker.internal:3001`. |

### Endpoints

| Service | URL |
|---|---|
| Postgres | `localhost:5432` |
| pgAdmin | <http://localhost:5050> |
| Auto Ontology ingestion service | <http://localhost:3002> |
| Auto Ontology frontend (full stack only) | <http://localhost:3000> |
| Auto Ontology backend (full stack only) | <http://localhost:3001> |

### `--dev` workflow

After `./dev_tools/setup_env.sh --dev` brings the infra up, start the apps locally in two terminals:

```bash
# terminal 1 — Next.js
cd frontend && pnpm install && pnpm dev

# terminal 2 — FastAPI (from the repo root)
uv sync
uv run uvicorn auto_ontology.server.__main__:create_app --factory --reload --host 127.0.0.1 --port 3001
```

The app factory is `create_app()` in `auto_ontology/server/__main__.py`; there is no
`auto_ontology/server/main.py`. `uv run python -m auto_ontology.server` starts the same app without
`--reload` — that is what the container runs.

### `--ds` workflow

`--ds` runs the Next.js frontend in docker but expects you to run the FastAPI backend locally — useful when you're iterating on backend Python code but don't care about rebuilding the frontend.

The frontend image is rebuilt with `PYTHON_API_URL=http://host.docker.internal:3001` baked into its Next.js rewrites, so requests from the browser hit the host's `:3001`.

```bash
bash ./dev_tools/setup_env.sh --ds

# then, on the host:
uv sync
uv run uvicorn auto_ontology.server.__main__:create_app --factory --reload --host 127.0.0.1 --port 3001
```

> Linux note: `host.docker.internal` works out-of-the-box on Docker Desktop (macOS / Windows). On native Linux Docker you may need to add `--add-host=host.docker.internal:host-gateway` or bind the backend to `0.0.0.0` and use the docker bridge IP.

### Configuration

Service ports and credentials come from `.env` at the repo root (read by `docker compose` automatically). See the root `README.md` and `CLAUDE.md` for the full variable list.

### Stopping

```bash
docker compose down       # stop everything
docker compose down -v    # stop + wipe volumes (Postgres, pgAdmin data)
```

## Container & Kubernetes deployment

This repo ships Dockerfiles and a Helm chart for running the stack in
Kubernetes:

- `Dockerfile` — backend (FastAPI / uvicorn). Sources NeMo-Retriever from a
  BuildKit named context, defaulting to a small committed stub.
- `frontend/Dockerfile` — frontend (Next.js standalone)
- `helm/auto-ontology/` — Helm chart with deployments, services, optional Ingress,
  HPA, PDB and a sample Postgres manifest
