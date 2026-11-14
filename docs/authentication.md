<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Authentication, SSO, and API tokens

Auto-Ontology authenticates through [Better Auth](https://github.com/better-auth/better-auth)
and authorizes every call against the caller's role. Self-service sign-up is
disabled: the first account is the bootstrap admin from
`AUTO_ONTOLOGY_ADMIN_EMAIL` / `AUTO_ONTOLOGY_ADMIN_PASSWORD` (Helm: `adminEmail` /
`adminPassword`).

Every `/api/...` route resolves the caller in one place,
[`frontend/auth/resolve-user.ts`](../frontend/auth/resolve-user.ts), in this
order:

1. Browser session cookie
2. Auto-Ontology API token (`x-api-key`, or `Authorization: Bearer`)
3. OAuth bearer issued by Auto-Ontology to an MCP client
4. SSO ID token forwarded by a trusted caller

## SSO

Configure SSO under **Settings → Single Sign-On**. Register this redirect URI
with the identity provider:

```text
<APP_URL>/api/auth/sso/callback
```

## API tokens

A browser gets a session cookie; scripts, notebooks, and scheduled jobs use an
API token instead.

1. Open the user menu (top right) → **API Tokens** → **New token**.
2. Name it, optionally set an expiry (default: never), and copy the token.
   Auto-Ontology stores only a SHA-256 hash, so the token is shown once. If you
   lose it, mint a new one.

Send it as `x-api-key` (or `Authorization: Bearer <token>`) on any `/api/...`
call:

```sh
curl -H "x-api-key: $AUTO_ONTOLOGY_API_TOKEN" https://auto-ontology.example.com/api/terms
```

```python
import os
import requests

session = requests.Session()
session.headers["x-api-key"] = os.environ["AUTO_ONTOLOGY_API_TOKEN"]
terms = session.get("https://auto-ontology.example.com/api/terms").json()
```

For the streaming question API, see
[conversation-api.md](./conversation-api.md).

How tokens behave:

- **A token acts as its owner.** It has no permissions of its own; each call is
  authorized against the owner's role, as in the browser. A viewer's token can
  do what a viewer can.
- **Every authenticated route accepts it.** Any operation in
  [`openapi/auto-ontology-api.json`](./openapi/auto-ontology-api.json) that
  requires authentication works with a token.
- **Revocation is immediate.** Delete the token, or delete or ban its owner,
  and the next request returns 401.
- **Tokens cannot manage tokens.** Creating and revoking tokens requires a
  signed-in session, so a leaked token cannot mint its own successors.
