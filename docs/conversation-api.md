<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Conversation API

Clients ask questions through the authenticated Next.js gateway:

```text
POST /api/chat/completions
```

Authenticate with a session cookie or an API token (see
[authentication.md](./authentication.md)). The request and response schemas are
in [`openapi/auto-ontology-api.json`](./openapi/auto-ontology-api.json).

## Response stream

The response is a server-sent event stream. Each event's `data:` line holds one
JSON object, and the stream ends with `data: [DONE]`, a final line that isn't
JSON.

| `type`   | Carries |
| -------- | ------- |
| `step`   | An agent step starting or ending (`node`, `phase`, and `thought` on end), for showing progress. |
| `sql`    | A query that passed validation and is about to run. |
| `result` | The final `answer`: `response` (prose), `sql_code`, `sql_response_from_db` (rows), `thoughts` (step trace). |
| `charts` | A chart or table for the result (`content`, `sql_response`). Sent after `result` when the query ran and the request includes a `conversation_id`. |
| `error`  | A `message` describing why the run failed. |

A run makes many sequential model calls and typically takes tens of seconds, so
read the stream as it arrives:

```python
import json
import os

import requests

response = requests.post(
    "https://auto-ontology.example.com/api/chat/completions",
    headers={"x-api-key": os.environ["AUTO_ONTOLOGY_API_TOKEN"]},
    json={"question": "How many orders shipped last week?"},
    stream=True,
    timeout=900,
)
response.raise_for_status()

for line in response.iter_lines(decode_unicode=True):
    if not line or not line.startswith("data:"):
        continue
    payload = line.removeprefix("data:").strip()
    if payload == "[DONE]":
        break
    event = json.loads(payload)
    if event["type"] == "step":
        print("…", event.get("node"), event.get("phase"))
    elif event["type"] == "result":
        answer = event["answer"]
        print(answer["response"])
        print(answer["sql_code"])
    elif event["type"] == "error":
        raise RuntimeError(event["message"])
```

The MCP server's `ask_question` tool wraps this same call and returns the
answer, SQL, and rows as separate fields; see [mcp.md](./mcp.md).

## Conversations

`conversation_id` is optional:

- Omit it for a stateless, one-shot request.
- Send a client-generated UUID to start or continue a conversation, and reuse it
  for every turn in the thread.
- Wait for `[DONE]` before sending the next turn. An overlapping request on the
  same conversation returns `409 Conversation in progress`.

```json
{
  "question": "What about last month?",
  "conversation_id": "d61d8aa3-d496-4fa7-97ce-4f831c162e7f"
}
```

For a conversational turn, the backend loads up to the five most recent
completed turns (capped at 12,000 characters) and rewrites a contextual
follow-up such as "What about August?" into a standalone question before
running the text-to-SQL agent. Independent questions pass through unchanged.
Only prior questions, answers, and SQL feed that rewrite; agent thoughts, chart
payloads, errors, and raw result rows stay out of it.

Conversation history belongs to the authenticated user. An API-created
conversation opens in the UI at `/chat?focus=<conversation_id>` for the user who
owns it. The web UI manages its own conversation IDs.

## Trust boundary

Call the Next.js gateway (`APP_URL`, port 3000 in Docker Compose). It resolves
the user and forwards an internal `x-auto-ontology-user-id` header to FastAPI
(port 3001). FastAPI trusts that header without verifying it, so keep port 3001
on a private network. A direct FastAPI request without `conversation_id` stays
stateless; one with `conversation_id` requires the trusted identity header.
