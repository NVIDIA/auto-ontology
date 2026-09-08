---
name: gsf-install
version: "0.1.0"
description: >-
  Install and troubleshoot a GSF (Generative Semantic Fabric) deployment via
  Docker Compose, local --dev, Helm, or MCP against an existing instance. Use
  when setting up GSF, .env or NIM keys fail, the UI looks broken after restart,
  Postgres ports collide, or MCP cannot sign in. For contributing to GSF
  internals see CLAUDE.md, not this skill.
license: Apache-2.0
metadata:
  author: NVIDIA GSF Team
  tags:
    - gsf
    - install
    - docker
    - helm
    - mcp
    - troubleshooting
---

# GSF Install

Bring GSF up, or connect an agent to an instance that is already running.
Do not invent a second installer: invoke the commands in the repository-root
`README.md` and `dev_tools/setup_env.sh`.

On failure, read [troubleshooting.md](references/troubleshooting.md) instead of
searching the web.

## Required questions

Ask these if not already clear. Do not guess a default and emit a command.

1. **Target** — full Docker Compose stack, local `--dev` (infra in Docker, apps
   on the host), `--ds` (frontend in Docker, FastAPI on the host), Helm /
   Kubernetes, or “GSF is already up, I only need MCP”?
2. **NVIDIA NIM key** — is `DEFAULT_MODELS_API_KEY` (or `NVIDIA_API_KEY`) set?
   Chat and ingest need it.
3. **Source database** — Postgres, Snowflake, Databricks, or DuckDB connection
   string for `CONNECTION_STRINGS`, or will connections be added in the UI?
4. **Port conflicts** — is host `5432` free? `POSTGRES_PORT` only remaps the
   host side; containers still use 5432 internally.

## Install logging (mandatory)

Wrap **every** install, compose, helm, and troubleshooting command with
[`scripts/log_install.sh`](scripts/log_install.sh) so the session is
debuggable. Pass the model id when the harness knows it:

```bash
# from the GSF repository root
./skills/gsf-install/scripts/log_install.sh --model "$GSF_INSTALL_MODEL" -- \
  ./dev_tools/setup_env.sh
```

The script appends timestamp, model, cwd, the command (with secret flags
redacted) and redacted env, and exit code to `.gsf-install.log` (gitignored). Do not
restate flags in this skill; log what actually ran.

## One-time prep (local)

From the repository root:

```bash
cp .env.example .env
# edit .env — AUTH_SECRET and APP_URL are required
```

`AUTH_SECRET` and `APP_URL` are required. Without them every page fails with a
Better Auth "default secret" error. Fill `DEFAULT_MODELS_API_KEY` and a
`CONNECTION_STRINGS` value (or plan to add connections in the UI). The full
variable list is `.env.example`.

For `--dev` / `--ds` you also need:

```bash
cd frontend && pnpm install   # Next.js
# from repo root:
uv sync
```

## Local: full Compose stack

Preferred when the user wants a running GSF without iterating on the code.

```bash
./skills/gsf-install/scripts/log_install.sh -- ./dev_tools/setup_env.sh
# equivalent: docker compose up -d --build
```

This builds `gsf` and `gsf-frontend`, starts Postgres, pgAdmin, the ingestion
service, runs the one-shot migrate job, and starts the app.

| Service | URL |
| --- | --- |
| UI | http://localhost:3000 |
| FastAPI (internal) | http://localhost:3001 |
| Ingestion | http://localhost:3002 |
| pgAdmin | http://localhost:5050 |
| Postgres | localhost:5432 |

## Local: `--dev` (apps on the host)

Infra only (Postgres, pgAdmin, ingestion). You run Next.js and FastAPI:

```bash
./skills/gsf-install/scripts/log_install.sh -- ./dev_tools/setup_env.sh --dev
```

Then two terminals:

```bash
cd frontend && pnpm dev
# repo root:
uv run uvicorn gsf.server.__main__:create_app --factory --reload --host 127.0.0.1 --port 3001
```

The app factory is `create_app()` in `gsf/server/__main__.py`. There is no
`gsf/server/main.py`. `uv run python -m gsf.server` is the same app without
reload (what the container runs).

## Local: `--ds` (frontend in Docker, API on the host)

```bash
./skills/gsf-install/scripts/log_install.sh -- ./dev_tools/setup_env.sh --ds
uv run uvicorn gsf.server.__main__:create_app --factory --reload --host 127.0.0.1 --port 3001
```

The frontend image bakes `PYTHON_API_URL=http://host.docker.internal:3001`.
On native Linux Docker you may need `--add-host=host.docker.internal:host-gateway`.

## Kubernetes

Do not paste Helm credentials into the conversation. Point at repository-root
`DEPLOYMENT.md` and ask the user for chart source, `defaultModelsApiKey`,
`postgresPassword`, and `connectionStrings`. This skill does not cover Astra
GitOps.

## GSF is already up — MCP only

Do not reinstall. Point `GSF_API_URL` at the **web app** (Compose UI is
`:3000`, not FastAPI `:3001`):

```bash
GSF_API_URL=http://localhost:3000 uvx --from "git+https://github.com/NVIDIA/GSF.git#subdirectory=mcp" gsf-mcp
```

Until the package is on PyPI this needs GitHub credentials that can read
`NVIDIA/GSF`. Client config uses the MCP server URL with a `/mcp` suffix.
People sign in through GSF; do not put a token on the MCP server.

Confirm the deployment is new enough to be an authorization server:

```bash
curl -s -o /dev/null -w '%{http_code}\n' "$GSF_API_URL/.well-known/oauth-authorization-server"
```

`200` is required. Anything else: run the frontend from the checkout
(`pnpm dev`) and point `GSF_API_URL` at that port.

## Verify

1. UI loads at http://localhost:3000 (or the deployed `APP_URL`) and shows
   the left navigation. Missing nav → stale cookie; see
   [troubleshooting.md](references/troubleshooting.md).
2. `GET /api/semantic-compilation/status` on the **web** origin authenticates
   (cookie, `x-api-key`, or SSO bearer) and returns JSON. `calculated: false`
   means the glossary is empty — ingest / compile, do not keep retrying chat.
3. MCP: OAuth discovery returns 200 (above). Then `check_readiness` once a
   client is connected.

## Stop

```bash
docker compose down          # keep volumes
docker compose down -v       # wipe Postgres / pgAdmin data
```

## See also

- [troubleshooting.md](references/troubleshooting.md)
- `gsf-agent` — call GSF once it is running
- `gsf-ontology` — edit the semantic layer
- `CLAUDE.md` — maintain GSF itself
