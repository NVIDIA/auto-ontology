---
name: gsf-agent
version: "0.1.0"
description: >-
  Scaffold how an agent calls GSF: prefer MCP for reads, REST for writes and
  when MCP is unavailable; discover datasets through the semantic layer; auth
  via MCP login, API token, or SSO bearer (AI-Q). Use when embedding GSF in an
  agent harness, AI-Q, NemoClaw, or another app, or when asking how to list
  terms, ask questions, or authenticate. Does not prescribe an agent framework.
license: Apache-2.0
metadata:
  author: NVIDIA GSF Team
  tags:
    - gsf
    - mcp
    - api
    - agents
    - aiq
---

# GSF Agent usage

Lightweight scaffolding for calling GSF from an agent. This is not a
prescribed harness, SDK, or LangGraph template.

If GSF is not running, use `gsf-install` first. If you need to **change**
terms or SQL attributes, use `gsf-ontology`.

## Prefer MCP for reads

When the harness speaks MCP, use it. The **handshake tool list** is the live
source of truth (`mcp/gsf_mcp/tools.py`). Do not copy a tool table into the
session; if the handshake and this skill disagree, trust the handshake.

Sequence (the server also advertises this so models do not jump straight to
`ask_question`):

1. `check_readiness` — compiled semantic layer **and** a live DB connection.
2. `search_terms` — what the nouns mean here.
3. `check_answerable` — cheap coverage check.
4. `ask_question` — full text-to-SQL; tens of seconds; answer + SQL + rows
   (capped at 100 rows, with `row_count` / `truncated`).

Everything on MCP **reads**. There is no MCP tool for raw databases, schemas,
or columns on purpose. `describe_table` is the legitimate table view: terms
and SQL attributes the table participates in.

Connect: `GSF_API_URL` is the **web app**. Run `gsf-mcp`, point the client at
`…/mcp`, user signs in. No token on the MCP server. Details:
repository `mcp/README.md` and `docs/mcp.md`.

## REST when MCP is unavailable, and for writes

Public clients call the authenticated **Next.js gateway**, not FastAPI
`:3001`. FastAPI trusts `x-gsf-user-id` and must not be exposed. A direct
FastAPI request without `conversation_id` is stateless; with
`conversation_id` it requires that internal header.

Auth (any of these; resolved in one place):

- Browser session cookie
- API token: `x-api-key: $GSF_API_TOKEN` (`Authorization: Bearer` works for
  `gsf_…` tokens too). Token acts as its owner.
- GSF-issued OAuth access token (MCP sign-in)
- SSO id token (`Authorization: Bearer <jwt>`), e.g. AI-Q — see
  [stack.md](references/stack.md)

Shapes: `docs/openapi/gsf-api.json`. Do not scrape all ~87 operations.

### Chat

`POST /api/chat/completions` — SSE `step` / `result` / `error` / `charts`,
then `[DONE]`. Permission `chat:use`.

- Omit `conversation_id` for a one-shot (no history, no chart step).
- Supply a client-generated UUID to create or continue a thread.
- Wait for `[DONE]` before the next turn; overlapping requests return
  **`409 Conversation in progress`**.
- A 404 on a conversation id means it belongs to another user.

Python (from the GSF README; no extra SDK):

```python
import os
import requests

session = requests.Session()
session.headers["x-api-key"] = os.environ["GSF_API_TOKEN"]

terms = session.get("https://gsf.example.com/api/terms").json()

answer = session.post(
    "https://gsf.example.com/api/chat/completions",
    json={"question": "How many orders shipped last week?"},
)
# response is SSE, not a single JSON object
```

Open an API-created thread in the UI at `/chat?focus=<conversation_id>` when
it belongs to the signed-in user.

### Discovery (semantic layer, not a catalog browser)

MCP first. REST fallback:

- `GET /api/terms` (`q`, `skip`, `limit`)
- `GET /api/terms/{term_id}`
- `GET /api/exploration/tables/{table_id}/details`
- `GET /api/exploration/graph`

Writes and compile reset: `gsf-ontology`.

## Empty or failed answers

`ask_question` / chat returned nothing useful → `check_readiness` (or
`GET /api/semantic-compilation/status`) before rewriting the question. A
named database is not proof SQL can execute. See `gsf-install`
troubleshooting.

## See also

- [stack.md](references/stack.md) — AI-Q, Nemotron, cuOpt (what exists vs
  what partners wire)
- `gsf-ontology` — modify the layer
- `gsf-install` — bring-up and MCP OAuth discovery
