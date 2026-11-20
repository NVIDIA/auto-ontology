<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Apache AGE graph projection (optional)

Auto-Ontology stores its catalog and semantic layer in plain Postgres tables.
This optional feature projects those rows into an openCypher graph with
[Apache AGE](https://age.apache.org/), inside the same database, so the
ontology can be traversed with Cypher and walked outward from a Term.

The tables stay the source of truth. The graph is derived, rebuilt in full,
and read-only for everything except the sync.

## Requirements

- A Postgres server with the AGE extension available. The `apache/age` image
  works; the compose file's `pgvector/pgvector` image does not ship AGE, so a
  deployment needs an image with both extensions to use vectors and the graph
  together.
- `AGE_GRAPH_ENABLED=true`. Optionally `AGE_GRAPH_NAME` (default
  `auto_ontology`).

The sync runs `CREATE EXTENSION IF NOT EXISTS age` and loads the library per
transaction, so `shared_preload_libraries` does not need to change.

## Graph model

Labels and edge types reuse the names the codebase already uses for the same
things (`auto_ontology/catalog/constants.py`,
`auto_ontology/semantic/constants.py`). The mapping is
`auto_ontology/graph/mapping.py`.

| Vertex | From table |
| --- | --- |
| `Database`, `Schema`, `Table`, `Column` | `catalog_*` |
| `Term` | `term` |
| `ColumnAttribute`, `SqlAttribute` | `column_attribute`, `sql_attribute` |
| `TermFamily` | `term_family` |
| `Concept`, `ConceptKey` | `concept`, `concept_key` |
| `Tag`, `Zone` | `tag`, `zone` |

| Edge | Shape | From table |
| --- | --- | --- |
| `CONTAINS` | Database→Schema→Table→Column, Schema→TermFamily, Database→Concept | parent FKs |
| `FOREIGN_KEY` | Column→Column | `column__foreign_key` |
| `JOIN` | Table→Table, Column→Column | `table__join`, `column__join` |
| `UNION` | Column→Column | `column__union` |
| `REPRESENTS` | Table→Term | `table__term` |
| `HAS_ATTRIBUTE` | Column→ColumnAttribute | `column__has_attribute` |
| `SEMANTIC_FK` | Column→ConceptKey (`use_in_join_paths` on the edge) | `column__semantic_fk` |
| `HAS_KEY` | Concept→ConceptKey | `concept_key.concept_id` |
| `KEY_COLUMN` | ConceptKey→Column, one per member table | `concept_key__column` |
| `PROPERTY_OF` | ColumnAttribute/SqlAttribute→Term | `*_attribute__term` |
| `MEMBER_OF` | Term→TermFamily (`base_name` on the edge), Term→Concept | `term__term_family`, `term__concept` |
| `TAGGED` | Tag→Term/Table/Column/attribute | `tag_target` |
| `ZONE_OF` | Zone→Database/Schema/Table | `zone_target` |

Every vertex carries its Auto-Ontology `id`. Connection settings
(`catalog_database.connection`) and sampled values
(`catalog_column.sample_values`) are never copied; a test enforces it.
Statement-tier rows (`sql_query`) are left out.

## Keeping it current

A sync rebuilds the graph in one `REPEATABLE READ` transaction, so the graph
matches a single snapshot of the catalog. It loads a staging graph of its own
(`<name>_<4 hex>`, drawn until no graph has that name, so a rebuild never
touches a graph it did not create) and only then drops the active graph and
renames the staging graph into its place. Readers keep reading the old graph
while the new one loads. Reads hold a shared advisory lock on the graph and the
swap takes the exclusive one, so the swap waits for reads in progress, and a
read that arrives during the swap waits for the commit and then reads the new
graph. Graph names are limited to 58 characters so the staging name still fits
Postgres' identifier limit, and `--batch-size` must be at least 1. It runs:

- automatically at the end of every semantic compilation pass, after tag rules
  are re-applied, when `AGE_GRAPH_ENABLED` is set. A failed sync is logged and
  does not fail the pass;
- on demand: `uv run python -m auto_ontology.graph sync`.

`synced_at` in the status output says when that was; edits made later appear
after the next sync.

## Reading it

| Surface | Call |
| --- | --- |
| CLI | `python -m auto_ontology.graph status` |
| CLI | `python -m auto_ontology.graph neighborhood <term_id> --depth 2` |
| CLI | `python -m auto_ontology.graph query "MATCH (t:Term)<-[:REPRESENTS]-(x:Table) RETURN t.name, x.name" --columns term,table` |
| REST | `GET /api/graph/status` |
| REST | `GET /api/graph/terms/{term_id}/neighborhood?depth=1..3&max_nodes=1..1000` |
| MCP | `explore_term_graph` |

The CLI `query` command and every read run in a `READ ONLY` transaction that is
rolled back, so a Cypher `CREATE` is refused. REST and MCP do not accept Cypher;
the neighbourhood walk is fixed server-side, one hop per statement, and stops at
`max_nodes` with `truncated: true`. Each hop is SQL over AGE's storage tables
through the `start_id`/`end_id` indexes the sync creates, rather than a Cypher
`MATCH (a)-[]-(b)`, which AGE plans as a scan of every vertex table. A hop reads
the ids of the frontier's edges, keeps no more new vertices than the cap still
allows, and fetches properties for those alone; edges between returned
vertices, the last hop's included, are all reported.

A vertex is identified by its `id` and `label` together: a Concept shares its
`id` with the Term it was created from. Edges in the neighbourhood carry
`source_label` and `target_label` for the same reason.

Both REST routes answer 503 while `AGE_GRAPH_ENABLED` is unset, or while the
server has no AGE extension (reads never create it; the first sync does).

## Testing

`auto_ontology/graph/tests/test_projection_age.py` is marked `age`. It runs
against a migrated database on an AGE-enabled server and skips otherwise. CI
runs it in its own job on `apache/age`, and the main suite deselects it with
`-m "not age"`:

```bash
docker run -d --name ao-age -e POSTGRES_PASSWORD=age -p 55432:5432 apache/age:release_PG16_1.6.0
export POSTGRES_HOST=localhost POSTGRES_PORT=55432 POSTGRES_USER=postgres POSTGRES_PASSWORD=age
psql -h localhost -p 55432 -U postgres -c 'create database auto_ontology'
uv run alembic upgrade head
uv run pytest -m age auto_ontology/graph
```
