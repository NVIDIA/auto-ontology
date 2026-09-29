---
name: nvidia-ontology-setup
version: "0.2.0"
description: >-
  Set up or troubleshoot the Auto Ontology runtime. Use for Docker Compose,
  local or Helm setup, and MCP connection to an existing deployment.
license: Apache-2.0
metadata:
  author: "NVIDIA <opensource@nvidia.com>"
  tags:
    - nvidia-ontology
    - install
    - docker
    - helm
    - mcp
    - troubleshooting
---

# Auto Ontology setup

## Purpose

Bring up the current Auto Ontology implementation, troubleshoot it, or connect an agent
to an instance that is already running. Do not invent a second installer:
invoke the commands in the repository-root `README.md` and
`dev_tools/setup_env.sh`. Typed setup, connection, ingestion, compilation, and
readiness artifacts are in
[runtime-contract.yaml](assets/runtime-contract.yaml).

On failure, read [troubleshooting.md](references/troubleshooting.md) instead of
searching the web.

## Instructions

### Required questions

Ask these if not already clear. Do not guess a default and emit a command.

1. **Target** — full Docker Compose stack, local `--dev` (infra in Docker, apps
   on the host), `--ds` (frontend in Docker, FastAPI on the host), Helm /
   Kubernetes, or “Auto Ontology is already up, I only need MCP”?
2. **NVIDIA NIM key** — is `DEFAULT_MODELS_API_KEY` (or `NVIDIA_API_KEY`) set?
   Chat and ingest need it.
3. **Source database** — Postgres, Snowflake, Databricks, or DuckDB connection
   string for `CONNECTION_STRINGS`, or will connections be added in the UI?
4. **Port conflicts** — is host `5432` free? `POSTGRES_PORT` only remaps the
   host side; containers still use 5432 internally.

## Available scripts

| Script | Purpose | Arguments |
| --- | --- | --- |
| `scripts/log_setup.sh` | Run and redact-log installation or troubleshooting commands | Optional `--model ID`, then `-- COMMAND [ARGS...]` |

## Setup logging (mandatory)

Invoke the script through the shell available to the agent. Wrap **every**
install, compose, helm, and troubleshooting command with
[`scripts/log_setup.sh`](scripts/log_setup.sh) so the session is
debuggable. Pass the model id when the harness knows it:

```bash
# from the Auto Ontology repository root
./skills/nvidia-ontology-setup/scripts/log_setup.sh --model "$NVIDIA_ONTOLOGY_SETUP_MODEL" -- \
  ./dev_tools/setup_env.sh
```

The script appends timestamp, model, cwd, the command (with secret flags
redacted) and redacted env, and exit code to `.nvidia-ontology-setup.log` (gitignored). Do not
restate flags in this skill; log what actually ran.

## Prerequisites for local installation

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

Preferred when the user wants a running Auto Ontology without iterating on the code.

```bash
./skills/nvidia-ontology-setup/scripts/log_setup.sh -- ./dev_tools/setup_env.sh
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
./skills/nvidia-ontology-setup/scripts/log_setup.sh -- ./dev_tools/setup_env.sh --dev
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
./skills/nvidia-ontology-setup/scripts/log_setup.sh -- ./dev_tools/setup_env.sh --ds
uv run uvicorn gsf.server.__main__:create_app --factory --reload --host 127.0.0.1 --port 3001
```

The frontend image bakes `PYTHON_API_URL=http://host.docker.internal:3001`.
On native Linux Docker you may need `--add-host=host.docker.internal:host-gateway`.

## Limitations

This skill uses the repository's existing installers and does not cover Astra
GitOps, change semantic definitions, or treat partial Vault configuration as
secret storage. Ask before destructive volume deletion.

## Kubernetes

Do not paste Helm credentials into the conversation. Point at repository-root
`DEPLOYMENT.md` and ask the user for chart source, `defaultModelsApiKey`,
`postgresPassword`, and `connectionStrings`. This skill does not cover Astra
GitOps.

## Auto Ontology is already up — MCP only

Do not reinstall. Point `GSF_API_URL` at the **web app** (Compose UI is
`:3000`, not FastAPI `:3001`):

```bash
GSF_API_URL=http://localhost:3000 uvx --from "git+https://github.com/NVIDIA/GSF.git#subdirectory=mcp" gsf-mcp
```

Until the package is on PyPI this needs GitHub credentials that can read
`NVIDIA/GSF`. Client config uses the MCP server URL with a `/mcp` suffix.
People sign in through Auto Ontology; do not put a token on the MCP server.

Confirm the deployment is new enough to be an authorization server:

```bash
curl -s -o /dev/null -w '%{http_code}\n' "$GSF_API_URL/.well-known/oauth-authorization-server"
```

`200` is required. Anything else: run the frontend from the checkout
(`pnpm dev`) and point `GSF_API_URL` at that port.

## Examples

- Full local stack: use `./dev_tools/setup_env.sh` through the logging wrapper.
- Existing deployment: do not reinstall; configure `GSF_API_URL` and verify
  OAuth discovery before connecting the MCP client.

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

## Troubleshooting

Use [troubleshooting.md](references/troubleshooting.md) for stale sessions,
model-key failures, embedding mismatches, port conflicts, and partial Vault
configuration. Log the command and redact secrets before sharing evidence.

## See also

- [troubleshooting.md](references/troubleshooting.md)
- [runtime-contract.yaml](assets/runtime-contract.yaml)
- `nvidia-ontology-query` — call Auto Ontology once it is running
- `nvidia-ontology-management` — edit the semantic layer
- `CLAUDE.md` — maintain Auto Ontology itself
