<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Auto Ontology MCP server

Lets any MCP-capable agent — Cursor, Claude Desktop, an internal agent — ask
questions about your data in natural language, and inspect the semantic layer
behind the answers. Anything that speaks [Model Context
Protocol](https://modelcontextprotocol.io) can use it.

## How it works

The server is an HTTP client of the public Auto Ontology API. It holds no database
credentials, no model configuration, and no credentials of its own:

```
agent harness  ──MCP──▶  auto-ontology-mcp  ──HTTPS──▶  Next.js (public API)
                                                   │
                                                   ▼
                                        FastAPI, Postgres
```

Authentication and permission checks already live in that Next.js layer, so the
server inherits them rather than reimplementing them. Two things follow: it can
run anywhere that can reach your deployment, and **a session can do exactly what
the person behind it can do** — a viewer gets a viewer's answers.

## Running it

Run one server; people log in through it. The URL of your Auto Ontology deployment is the
only thing it needs to be told:

```sh
AUTO_ONTOLOGY_API_URL=https://auto_ontology.example.com auto-ontology-mcp
```

Install with `uvx`, which fetches and builds straight from the repository, so
there is nothing to clone:

```sh
uvx --from "git+https://github.com/NVIDIA/auto-ontology.git#subdirectory=mcp" auto-ontology-mcp
```

> [!NOTE]
> Auto Ontology is NVIDIA-internal today, so this install needs GitHub credentials with
> access to the repository (`gh auth login`, or any cached git credential
> helper). It shortens to a plain `uvx auto-ontology-mcp` once the package is published to
> PyPI; until then `uvx auto-ontology-mcp` alone will not resolve.

## Connecting a client

Point the client at the URL. There are no credentials in the config:

```json
{
  "mcpServers": {
    "auto-ontology": {
      "url": "https://auto-ontology-mcp.example/mcp"
    }
  }
}
```

Note the `/mcp` suffix: that is the endpoint, not the server's root. Restart the
client after editing its config, since most read MCP configuration only at
startup.

The client sees that the server wants authorization and offers to sign in —
Cursor lists it as needing login. The user gets Auto Ontology's ordinary login page, with
whatever SSO that deployment uses, and the client ends up holding a token it
manages itself. Nothing is minted or pasted by hand, and every call runs as the
person who signed in.

Then ask something like *"what does Auto Ontology mean by an active customer, and how many
were there last quarter?"*

## Sign-in

Auto Ontology is the authorization server. This server holds no client id, no client
secret, and no redirect URI, so there is nothing to configure per deployment and
nothing to register with an identity provider — clients register themselves with
Auto Ontology automatically.

Registrations and grants live in Auto Ontology's database, so replicas share them and a
restart signs nobody out. Permissions come from the account on every call rather
than from the token, so changing a role, banning a user, or deleting a grant
takes effect immediately.

The tokens are opaque, so the server asks Auto Ontology's OAuth UserInfo endpoint to check
each one. That bearer-authenticated endpoint needs no client secret and checks
expiry, revocation, client, and browser-session state on every call. The grant
must include the `openid` scope (Better Auth's UserInfo requirement); Auto Ontology
advertises it on the authorization server so clients that copy
`scopes_supported` include it. If Auto Ontology cannot be reached the call fails as an
error rather than as "sign in again", so an outage does not send everyone into a
login that cannot succeed either.

There is no way to configure a token for the server itself. One would make every
caller act as its owner — their permissions, their conversation history — and
nothing in the protocol would reveal it.

## Pointing at a Auto Ontology on your own machine

Only `AUTO_ONTOLOGY_API_URL` changes. For the Docker stack the web app is on `:3000`:

```sh
AUTO_ONTOLOGY_API_URL=http://localhost:3000 auto-ontology-mcp
```

Sign-in needs a Auto Ontology new enough to be an authorization server, which an
already-built image may not be. Checking first is worth it, because otherwise the
failure surfaces as a client that cannot log in:

```sh
curl -s -o /dev/null -w '%{http_code}\n' $AUTO_ONTOLOGY_API_URL/.well-known/oauth-authorization-server
```

A `200` is what you want. Anything else means running the frontend from your
checkout instead — `pnpm dev`, with `AUTO_ONTOLOGY_API_URL` pointing at its port.

## Tools

Everything here reads. Nothing modifies the glossary, the catalog, or the
underlying databases.

| Tool | What it is for |
| --- | --- |
| `ask_question` | **The primary tool.** Ask a question in plain language; get the answer, the SQL Auto Ontology ran, and the rows. |
| `check_answerable` | Grade whether the semantic layer covers a question. Cheap pre-flight before `ask_question`. |
| `check_readiness` | Whether this deployment can answer anything at all. |
| `search_terms` | Search the business glossary. |
| `get_term` | One term: description, synonyms, related terms. |
| `get_term_columns` | The physical columns a term maps to. |
| `get_term_sql_attributes` | The SQL attributes defined under a term. |
| `get_sql_attribute` | One SQL attribute: its expression and purpose. |
| `describe_table` | A table's columns, related terms, and SQL attributes together. |

There is deliberately no tool for browsing databases, schemas, or raw columns.
Consumers are meant to reach the data through the semantic layer, and a catalog
browser invites an agent to reason about raw tables instead. `describe_table`
covers the legitimate case, since what it returns is the terms and SQL attributes
a table participates in.

The server also advertises **instructions** at handshake describing how the tools
sequence, which spares the model from inferring it — left to itself it tends to
reach straight for `ask_question`.

### About `ask_question`

It is a call to `POST /api/chat/completions`, the same endpoint the web UI uses,
which runs the full text-to-SQL agent: many sequential model calls, typically
tens of seconds. It returns the answer, the SQL, and the rows as separate fields
rather than the markdown the UI receives, so an agent can use the SQL without
parsing prose. Results are capped at 100 rows, with `row_count` and `truncated`
reporting what was withheld. It emits MCP progress notifications as the agent
works, mirroring the reasoning trace the web UI shows.

### About `check_readiness`

Answering a question needs two independent things: a compiled semantic layer to
resolve the question against, and a live database connection to run the SQL on.
Having one without the other is this deployment's most misleading state, because
everything looks healthy — the glossary reads, terms resolve — and every question
still fails, expensively and vaguely, after minutes of model calls.
`check_readiness` tells that apart up front, and reports every blocker at once.

Note that `databases` can be non-empty while `can_execute_sql` is false: the
catalog outlives the connection it was ingested from, so a named database is not
evidence that anything can be queried. Reading connections is also admin-only by
default, so a viewer gets a 403 there while chat and the catalog read fine — that
is neither a broken deployment nor a bad credential, so the fact is reported
under `unverified` and the verdict stands on what could be checked.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `AUTO_ONTOLOGY_API_URL` | `http://localhost:3000` | Base URL of the Auto Ontology web app. |
| `AUTO_ONTOLOGY_MCP_PUBLIC_URL` | `http://localhost:3003` | Public MCP origin, without `/mcp`. The frontend must receive the same value so issued tokens have the resource URL clients use. |

Rarely needed: `AUTO_ONTOLOGY_MCP_HOST` and `AUTO_ONTOLOGY_MCP_PORT` (bind address and port, default
`0.0.0.0:3003`), `AUTO_ONTOLOGY_MCP_TIMEOUT_S` and `AUTO_ONTOLOGY_MCP_CHAT_TIMEOUT_S` (request
timeouts, `30` and `900`), and `AUTO_ONTOLOGY_OPENAPI_SPEC` (override the spec tools are
generated from).

## Extending the tool surface

The tool set is an explicit allow-list in
[`mcp/auto_ontology_mcp/tools.py`](../mcp/auto_ontology_mcp/tools.py). The spec publishes 87
operations, and exposing all of them would degrade tool selection badly. To
publish another, add a `ToolSpec` naming its method, path, an agent-facing name,
and a description that says *when to reach for it*. Startup fails loudly if a
curated entry no longer exists in the spec, so a rename upstream cannot silently
drop a tool.

Parameters and descriptions come from
[`auto-ontology-api.json`](./openapi/auto-ontology-api.json), which is committed twice: canonically
under `docs/openapi/`, and again inside the package, because the server reads it
at startup and must work from an install where no `docs/` directory exists.
`pnpm openapi` writes both and CI diffs both, so never edit the packaged copy by
hand.

## Troubleshooting

**A client cannot discover how to sign in** — check that
`$AUTO_ONTOLOGY_API_URL/.well-known/oauth-authorization-server` returns JSON. A redirect to
the login page instead means the deployment predates that route.

**"Protected resource ... does not match expected ..."** — the server advertises
the address it was bound to, and the client is reaching it under a different
spelling of the same host, almost always `127.0.0.1` against `localhost`. Set
`AUTO_ONTOLOGY_MCP_PUBLIC_URL` to the URL the client uses, or leave `AUTO_ONTOLOGY_MCP_HOST` unset.

**"This request carried no signed-in session"** — the grant behind the call
expired or was revoked in Auto Ontology. Sign in again from the client.

**`ask_question` returned an empty answer, with no SQL and no error** — the run
completed but retrieval found nothing to build a query from. Call
`check_readiness` first: most often no database connection is configured, so no
question can succeed however it is phrased. If the deployment is ready, the
question's vocabulary is the problem — `search_terms` for the nouns in it.

**"Auto Ontology cannot answer right now"** — usually the semantic layer was never
compiled; confirm with `check_readiness`. It also appears when a
`conversation_id` already has a turn in flight.

**Client shows no tools** — check the client's MCP logs. The server logs to
stderr.
