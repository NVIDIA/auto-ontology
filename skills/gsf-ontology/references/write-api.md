<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Ontology write API (index)

Canonical request/response shapes: `docs/openapi/gsf-api.json`. Do not paste
schemas here. Permissions are `x-gsf-permissions` on each operation.

Call the **Next.js** origin (`APP_URL`, local `:3000`), not FastAPI `:3001`.

## Terms

| Method | Path | Permission | Notes |
| --- | --- | --- | --- |
| GET | `/api/terms` | `catalog:read` | Query `q`, `skip`, `limit` |
| GET | `/api/terms/{term_id}` | `catalog:read` | |
| PATCH | `/api/terms/{term_id}` | `catalog:edit` | `TermUpdate`: `name`, `description`, `name_certified`, `description_certified`. Renaming invalidates cached SQL-attribute description suggestions. |

There is no public "create empty term" POST on `/api/terms`. New terms arrive
through compilation, ingest, or model import.

## Column attributes

| Method | Path | Permission | Notes |
| --- | --- | --- | --- |
| GET | `/api/terms/{term_id}/column-attributes` | `catalog:read` | Physical columns a term maps to |
| PATCH | `/api/terms/{term_id}/column-attributes/{attr_id}` | `catalog:edit` | Name/description/certified/sample_values. `{attr_id}` is a ColumnAttribute id, not a SqlAttribute id. Refreshes the semantic embedding. |

## SQL attributes

| Method | Path | Permission | Notes |
| --- | --- | --- | --- |
| POST | `/api/sql-attributes/validate` | `catalog:edit` | Body: `expression`, optional `term_id`, optional `attribute_id`. Parse failure is **422**, not `valid: false`. |
| POST | `/api/sql-attributes` | `catalog:edit` | `SqlAttributeCreate`: required `name`, `description`, `expression`, `term_id`; optional `source` (default `manual`). 201. |
| GET | `/api/sql-attributes/{attr_id}` | `catalog:read` | Also MCP `get_sql_attribute` |
| PATCH | `/api/sql-attributes/{attr_id}` | `catalog:edit` | Metadata only (name/description). Does **not** change the SQL expression. |
| PUT | `/api/sql-attributes/{attr_id}` | `catalog:edit` | Re-parses SQL and refreshes the embedding. Same required fields as create. |
| DELETE | `/api/sql-attributes/{attr_id}` | `catalog:edit` | Graph cleanup plus embedding deletion |
| GET | `/api/terms/{term_id}/sql-attributes` | `catalog:read` | List under one term |
| GET | `/api/sql-attributes/{attr_id}/description-suggestion` | `catalog:edit` | LLM suggestion for the edit flow |

## Catalog node descriptions

| Method | Path | Permission | Notes |
| --- | --- | --- | --- |
| PATCH | `/api/nodes/{node_id}` | `catalog:edit` | Mutable properties of a Database, Schema, Table, or Column node |

## Compilation

| Method | Path | Permission | Notes |
| --- | --- | --- | --- |
| GET | `/api/semantic-compilation/status` | `chat:use` | Whether any Term exists (`calculated`) |
| POST | `/api/semantic-compilation/reset` | `semanticCompilation:manage` | **Destructive.** Deletes every database's compiled semantic layer. 202. Not a cleanup step after a small edit. |

## Model interchange

| Method | Path | Permission | Notes |
| --- | --- | --- | --- |
| POST | `/api/model/export` | `modelInterchange:export` | JSON body `ExportRequest`: `databases` (empty = all), `format` `gsf` \| `ossie`. Response is YAML. |
| POST | `/api/model/import` | `modelInterchange:import` | Multipart YAML. Query `replace` (default true), `embed` (default true). Native GSF vs Ossie is detected from the document root (`data_layer`/`semantic_layer` vs `semantic_model`). |

## Exploration (read; lineage)

All `catalog:read`:

- `GET /api/exploration/tables/{table_id}/details`
- `GET /api/exploration/terms/{term_id}/details`
- `GET /api/exploration/terms/{term_id}/path/{other_term_id}`
- `GET /api/exploration/graph`
- `GET /api/exploration/semantic-graph`
- `GET /api/exploration/edges`
- `GET /api/exploration/nodes/{node_id}/relationships`

## Verify after a write

Prefer MCP `check_answerable` then `ask_question`. REST equivalents:

- `POST /api/question-entity-coverage`
- `POST /api/chat/completions` (SSE; `chat:use`)
