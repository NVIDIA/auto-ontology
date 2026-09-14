# GSF

Generative Semantic Fabric adds the structured-data ontology layer to any partner or NVidia agent harness  interface, like NVIDIA AI-Q Claws, etc

> **Licensing & contributions.** GSF is distributed under the
> [Apache License 2.0](./LICENSE). Third-party open-source components
> bundled, linked, or otherwise used by this project are listed in
> [`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md). **This project
> is currently not accepting external contributions.**

<p>
<img src="./docs/assets/arch.png" alt="GSF Architecture" width="800">
</p>

## Key Features

- **Natural-language querying** of structured data — questions are translated to
  SQL and executed against connected databases, powered by NVIDIA NIM.
- **Semantic catalog / ontology layer** over relational sources, stored in
  Postgres alongside the catalog it describes.
- **Authentication & RBAC** —using [Better Auth](https://github.com/better-auth/better-auth).
- **Web UI** for chat, analysis, data catalog, analytics, and settings.
- **Flexible deployment** — Helm chart for Kubernetes or Docker Compose for a
  local stack.

## Software Components

| Component           | Technology                                                             | Role                                       | Default port |
| ------------------- | ---------------------------------------------------------------------- | ------------------------------------------ | ------------ |
| Frontend            | Next.js 16 (React 19, TypeScript, Tailwind CSS 4), Better Auth, Prisma | Web UI, authentication, API gateway        | 3000         |
| Backend             | FastAPI (Python 3.12+), NeMo-Retriever, LangChain                      | Chat / NL-to-SQL, catalog, datasource APIs | 3001         |
| Ingestion worker    | FastAPI (Python 3.12+), NeMo-Retriever                                 | Ingests tabular data and writes embeddings | 3002         |
| Postgres + pgvector | Relational Database                                                    | App metadata, the catalog and ontology, vector store | 5432 |
| HashiCorp Vault     | Optional                                                               | Secure storage of connection credentials   | —            |

### NVIDIA NIM

GSF uses NVIDIA NIM endpoints for inference — either the hosted NVIDIA
inference API ([inference-api.nvidia.com](https://inference-api.nvidia.com)) or self-hosted NIMs:

- **LLM:**
  [aws/anthropic/bedrock-claude-opus-4-8](https://inference.nvidia.com/aws/anthropic/bedrock-claude-opus-4-8?search=opus)
- **Embeddings:**
  [nemotron-3-embed-1b](https://inference.nvidia.com/nvidia/nvidia/nemotron-3-embed-1b)

The endpoints and models are configured via the `DEFAULT_MODELS_ENDPOINT`, `DEFAULT_MODELS_MODEL`,
`EMBED_ENDPOINT`, and `EMBED_MODEL` environment variables and require a
`DEFAULT_MODELS_API_KEY`.

#### Model triplets

Each model role is configured by a **triplet** of environment variables —
`<PREFIX>_ENDPOINT` (URL), `<PREFIX>_API_KEY`, and `<PREFIX>_MODEL`:

| Prefix           | Role                                                        |
| ---------------- | ----------------------------------------------------------- |
| `DEFAULT_MODELS` | Shared default; every field below falls back to it.         |
| `REASONING`      | Main chat / NL-to-SQL model.                                |
| `NON_REASONING`  | Lighter model used for entity extraction.                   |
| `EMBED`          | Text-embedding model (must match between ingest and query). |
| `RERANK`         | Reranker used by the retrieval flow.                        |

You can set a full triplet, only some of its fields, or none at all — any field
left unset falls back to the matching `DEFAULT_MODELS_<FIELD>`. So the simplest
setup is to fill in `DEFAULT_MODELS_*` and override per-triplet fields only where
they differ (e.g. `EMBED_MODEL`, `NON_REASONING_MODEL`). `*_API_KEY` also falls
back to the legacy `NVIDIA_API_KEY` for backward compatibility.

## Deployment

### Prerequisites

- **Docker** with **Docker Compose v2** (e.g. Docker Desktop on macOS/Windows,
  Docker Engine on Linux) for the local stack. For Kubernetes deployments see
  [`DEPLOYMENT.md`](./DEPLOYMENT.md).
- An **NVIDIA API key** for NVIDIA NIM (chat and ingestion) from the NVIDIA
  inference API at <https://inference-api.nvidia.com>.
- Connection details for the source database(s) you want to query
  (Databricks, Postgres, Snowflake, or DuckDB).

### Installation

**Kubernetes**

To deploy GSF on a Kubernetes cluster, see [`DEPLOYMENT.md`](./DEPLOYMENT.md).

**Local (Docker Compose)**

1. Clone the repository:

   ```bash
   git clone <repo-url> gsf && cd gsf
   ```

2. Create your environment file (.env) from the template and fill in the values
   (Postgres credentials, `DEFAULT_MODELS_API_KEY`, `CONNECTION_STRINGS`, etc.).
   See [`.env.example`](./.env.example) for the full list of variables:

   ```bash
   cp .env.example .env
   # edit .env
   ```

   Note:
   - `AUTH_SECRET` and `APP_URL` are **required** — without them every page
     fails with a Better Auth "default secret" error.
   - `POSTGRES_PORT` only changes the **host-side** port mapping (set it if
     5432 is already taken on your machine); containers always reach Postgres
     on the internal network port.

3. Build the images and start the stack:

   ```bash
   docker compose up -d --build
   ```

   This builds the backend (`gsf`) and frontend (`gsf-frontend`) images, brings
   up Postgres and pgAdmin, runs the one-shot `frontend-migrate` job to
   sync the database schema, and starts the app (backend, ingestion service,
   and frontend).

4. Open the UI at <http://localhost:3000> (the backend API is on `:3001`,
   pgAdmin on `:5050`).

   > **Troubleshooting:** if the UI loads without the left navigation panel
   > (or otherwise looks broken after a restart), your browser is holding a
   > stale session cookie — this happens when `AUTH_SECRET` changes or the
   > database is reset. Clear the site data for `localhost:3000` (or delete
   > the `better-auth.session_token` cookie) and sign in again.

## Connections Management

GSF resolves the source databases it connects to from two sources:

1. **Added through the UI** — connections created from within the app. With this
   option the connection credentials are **stored in plaintext in Postgres unless
   Vault is configured**:
   - **Without Vault:** the full connection object (including the password) is
     JSON-serialized and stored, unencrypted, on the catalog's database row.
   - **With Vault:** the credentials are written to Vault, keyed by the database
     name, and the catalog row stores no credentials (the database name is the
     lookup key — there is no separate reference field).

   Vault is enabled only when all of `VAULT_ADDR`, `VAULT_NAMESPACE`,
   `VAULT_ROLE_ID`, and `VAULT_SECRET_ID` are set (`VAULT_AUTH_MOUNT` and
   `VAULT_KV_MOUNT` are optional overrides). It is off by default; a partial
   configuration is ignored with a warning and falls back to plaintext storage.

2. **`CONNECTION_STRINGS` environment variable** — a comma-separated list of
   connection strings supplied at deploy time (e.g. the Helm
   `--set connectionStrings=<CONNECTION-STRINGS>` flag). This is a fallback: it
   is used only when there are no UI-added connections.

## Authentication

GSF Supports SSO for authentication.
The Redirect URI should be configured in the IdP as: APP_URL/api/auth/sso/callback

### API tokens (scripting)

A browser signs in and gets a session cookie, which a script cannot obtain. For
scripts, notebooks, and scheduled jobs, mint an **API token** instead:

1. Open the user menu (top right) → **API Tokens** → **New token**.
2. Name it, optionally pick an expiry (default: never), and **copy the token**.
   GSF stores only a SHA-256 hash of it, so it is shown exactly once. Losing it
   means minting a new one.

Send it as `x-api-key` on any `/api/...` call — `Authorization: Bearer <token>`
works too:

```sh
curl -H "x-api-key: $GSF_API_TOKEN" https://gsf.example.com/api/terms
```

```python
import os
import requests

session = requests.Session()
session.headers["x-api-key"] = os.environ["GSF_API_TOKEN"]

terms = session.get("https://gsf.example.com/api/terms").json()

answer = session.post(
    "https://gsf.example.com/api/chat/completions",
    json={"question": "How many orders shipped last week?"},
).json()
```

Things worth knowing:

- **A token acts as its owner.** It carries no permissions of its own — every
  call is authorized against the owner's role, exactly as it would be in the
  browser. A viewer's token cannot do admin things.
- **The whole API surface accepts it.** Authentication is resolved in one place
  (`frontend/auth/resolve-user.ts`) for every route, so any endpoint in
  [`docs/openapi/gsf-api.json`](./docs/openapi/gsf-api.json) that is not
  `withPublic` works with a token.
- **Revocation is immediate.** Delete the token in the UI, or delete/ban the
  owning user, and the next request with it gets a 401.
- **Tokens cannot manage tokens.** Creating and revoking requires a signed-in
  session, so a leaked token cannot issue itself successors.
  
## Agent API conversations

Public clients call the authenticated Next.js gateway at
`POST /api/chat/completions`. The response is an SSE stream containing `step`,
`result`, or `error` events, followed by `[DONE]`.

`conversation_id` is optional:

- Omit it for a stateless, one-shot request.
- Supply a client-generated UUID to create or continue a conversation.
- Reuse the same UUID for every turn in the thread. Wait for `[DONE]` before
  sending the next turn; overlapping requests for one conversation return
  `409 Conversation in progress`.

```json
{
  "question": "What about last month?",
  "conversation_id": "d61d8aa3-d496-4fa7-97ce-4f831c162e7f"
}
```

For conversational requests, FastAPI loads up to the five most recent completed
turns (bounded to 12,000 characters) and rewrites contextual input such as
“What about August?” into a standalone question before running the existing
text-to-SQL graph. Independent questions are passed through unchanged. Internal
agent thoughts, chart payloads, errors, and raw SQL result sets are not reused as
context.

Conversation history is scoped to the authenticated GSF user. Browser and
service callers should use the Next.js gateway, which resolves the user and
forwards the internal `x-gsf-user-id` header to FastAPI. The FastAPI port is a
trusted internal service boundary: it must not be exposed publicly because that
header is not independently verified by FastAPI. A direct FastAPI request
without `conversation_id` remains stateless; a direct request with
`conversation_id` requires the trusted identity header.

The web UI manages the UUID automatically. An API-created conversation can be
opened in the UI at `/chat?focus=<conversation_id>` when it belongs to the
signed-in user.

## Agent skills

Coding agents: start at [`AGENTS.md`](./AGENTS.md). Skills live in
[`skills/`](./skills/) (`nvidia-ontology-install`, `nvidia-ontology`, `nvidia-ontology-agent`).
Maintainers keep using [`CLAUDE.md`](./CLAUDE.md).

## MCP server

Agent harnesses such as Cursor and Claude Desktop can query your data through
GSF's [MCP](https://modelcontextprotocol.io) server. See
[`mcp/`](./mcp/) to run it and [`docs/mcp.md`](./docs/mcp.md) for the details.

## License

GSF is licensed under the [Apache License, Version 2.0](./LICENSE).
SPDX identifier: `Apache-2.0`.

Each NVIDIA-authored source file in this repository carries an SPDX header
of the form:

```text
SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
```

Third-party open-source components used by GSF are enumerated, with their
upstream licenses and project URLs, in
[`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md).

## Contributing

**This project is currently not accepting contributions.** Issues, pull
requests, and patches submitted from outside the GSF maintainer team will
not be reviewed or merged. Security-relevant reports should follow the
process described in [`SECURITY.md`](./SECURITY.md).
