---
name: gsf-ontology
version: "0.1.0"
description: >-
  Inspect and modify the GSF semantic layer: glossary terms, SQL attributes,
  column attributes, term lineage, model YAML import/export, and semantic
  compilation. Use when editing ontology meaning, adding derived metrics,
  validating SQL attributes, or navigating what a dataset means. MCP is
  read-only; writes go through the REST API. Does not prescribe a modeling
  methodology.
license: Apache-2.0
metadata:
  author: NVIDIA GSF Team
  tags:
    - gsf
    - ontology
    - glossary
    - sql-attributes
    - catalog
---

# GSF Ontology

Scaffold for inspecting and changing GSF's semantic layer. This is not a
prescribed modeling methodology — propose the change the user asked for, using
the APIs below.

MCP tools only **read**. Create, patch, import, and compile-reset go through
the Next.js `/api/...` gateway. Request bodies and field types live in
`docs/openapi/gsf-api.json`; this skill names operations and permissions
only. See [write-api.md](references/write-api.md).

## Auth

Scripts use an API token (`x-api-key` or `Authorization: Bearer`). Mint it in
the UI (user menu → API Tokens). A token **acts as its owner** — a viewer's
token cannot do admin things. Creating and revoking tokens requires a signed-in
session; a token cannot mint another token.

Writes below need `catalog:edit` unless noted. `403` means the owner's role
lacks that permission, not that the path is wrong.

## Workflow

1. **Discover current meaning** via MCP (`search_terms`, `get_term`,
   `get_term_columns`, `get_term_sql_attributes`, `describe_table`) or REST
   if MCP is absent (`GET /api/terms`, `GET /api/terms/{term_id}`,
   `GET /api/exploration/tables/{table_id}/details`).
2. **State the proposed change** to the user (term rename, new SQL attribute,
   import, and so on). Do not silently rewrite the glossary.
3. **Validate SQL** before create/update: `POST /api/sql-attributes/validate`.
   A parse failure is **HTTP 422**, not `valid: false`.
4. **Apply** the smallest write that matches the request (see
   [write-api.md](references/write-api.md)).
5. **Compilation is not a hidden side effect.** Check
   `GET /api/semantic-compilation/status`. Do **not** call
   `POST /api/semantic-compilation/reset` as cleanup — it deletes every
   database's compiled layer, returns 202, and requires
   `semanticCompilation:manage`.
6. **Verify** with MCP `check_answerable` / `ask_question` (or REST
   `POST /api/question-entity-coverage` and `POST /api/chat/completions`).

## Lineage and "what does this dataset mean"

Stay on the semantic layer. Do not browse raw schemas to answer meaning.

- Term → columns: MCP `get_term_columns` or
  `GET /api/terms/{term_id}/column-attributes`
- Term → derived SQL: MCP `get_term_sql_attributes`
- Table → terms and SQL attributes: MCP `describe_table` or
  `GET /api/exploration/tables/{table_id}/details`
- Hop chain between two terms:
  `GET /api/exploration/terms/{term_id}/path/{other_term_id}`
- Graphs: `GET /api/exploration/graph`, `GET /api/exploration/semantic-graph`

## Bulk import / export

- `POST /api/model/export` — YAML of catalog + semantic layer.
  Permission `modelInterchange:export`. Body may set `databases` (empty =
  all) and `format` (`gsf` or `ossie`).
- `POST /api/model/import` — multipart YAML; native GSF
  (`data_layer` / `semantic_layer`) or Apache Ossie (`semantic_model`).
  Query `replace` (default true) and `embed` (default true). Permission
  `modelInterchange:import`. Can replace existing data — confirm with the
  user before `replace=true`.

## See also

- [write-api.md](references/write-api.md)
- `gsf-agent` — how to call GSF (MCP vs REST)
- `gsf-install` — deployment not ready
- `mcp/gsf_mcp/tools.py` — live read-tool allow-list
