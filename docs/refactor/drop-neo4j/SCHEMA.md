# Drop Neo4j — Postgres schema reference

Living, human-readable description of the `gsf` Postgres schema. **Regenerated
whenever a migration lands**, so nobody has to read Alembic revisions in order
to know what the tables look like.

Design rationale lives in [PLAN.md § Schema design](PLAN.md#schema-design) —
this file describes what *is*, not why. The metadata itself is
`gsf/dal/pg/schema.py`, which Alembic autogenerates from.

**Status: landed in Phase 3** (revision `75bdf1cdf36d`). 23 tables + the
`alembic_version` bookkeeping table, and one view. Not yet written to — Phase 4
ports the catalog write path onto it.

```bash
make migrate         # alembic upgrade head
make migrate-check   # fail if the database has drifted from schema.py
```

---

## Conventions

- Namespace **`gsf`** — never `public` (Prisma's) or `vdb`
  (langchain_postgres'). Set via `MetaData(schema="gsf")`, never `search_path`,
  which is per-session and would be a silent hazard on a pooled connection.
- Primary keys are `text` UUIDs, `DEFAULT (gen_random_uuid())::text`.
  Parenthesised to match how Postgres stores the expression, so
  `alembic check` doesn't report phantom drift on every id column.
- Catalog tier is prefixed `catalog_`; semantic tier uses the bare lowercased
  singular label.
- `CONTAINS` is a **parent FK column, not an edge table** — that is what turns
  the old APOC cascade into `ON DELETE CASCADE`.
- Every entity that model-interchange can import carries `imported_id`
  (indexed, non-unique), because `_resolve_entities_batch` matches
  `imported_id OR id`.

## Catalog tier

| Table | Cols | Replaces | Notes |
|---|---|---|---|
| `catalog_database` | 7 | `:Database` | `connection jsonb` — populated only when Vault is unconfigured |
| `catalog_schema` | 4 | `:Schema` | FK → `catalog_database`; unique on (database, name) |
| `catalog_table` | 7 | `:Table` | FK → `catalog_schema`; `pk text[]`, `table_type` |
| `catalog_column` | 10 | `:Column` | FK → `catalog_table`; `sample_values` kept as `text` (a JSON string today — promoting it to `jsonb` is a deliberate follow-up, not a porting detail) |
| `column_foreign_key` | 2 | `:FOREIGN_KEY` | Column → Column |
| `table_join` | 3 | `:JOIN` | Table → Table, carries `join_columns jsonb` |
| `sql_query` | 5 | `:Sql` | unique on `md5(sql_full_query)` — statement text can exceed the btree row limit, so the hash carries the constraint |
| `sql_query_table` | 2 | `:SQL` | Sql → Table |

## Semantic tier

| Table | Cols | Replaces | Notes |
|---|---|---|---|
| `term` | 8 | `:Term` | `synonyms text[]`, both certification flags; unique on (name, source) |
| `table_term` | 2 | `:REPRESENTS` | Table → Term |
| `column_attribute` | 11 | `:ColumnAttribute` | the 5-part merge key is a unique constraint; **`table_id` is plain `text` with no FK** — the importer legitimately writes `''` |
| `column_attribute_term` | 2 | `:PROPERTY_OF` | ColumnAttribute → Term |
| `column_attribute_link` | 3 | `:HAS_ATTRIBUTE` ∪ `:SEMANTIC_FK` | discriminated by `kind`, CHECK-constrained. One table because `find_join_path` traverses both together |
| `sql_attribute` | 8 | `:SqlAttribute` | `source` CHECK-constrained to manual/sql/table/bridgeTable |
| `sql_attribute_term` | 2 | `:PROPERTY_OF` | SqlAttribute → Term |
| `sql_attribute_sql` | 2 | `:HAS_SQL` | SqlAttribute → Sql |
| `custom_analysis` | 4 | `:CustomAnalysis` | |
| `custom_analysis_sql` | 2 | `:HAS_SQL` | CustomAnalysis → Sql |
| `pql_analysis` | 5 | `:PqlAnalysis` | never attached to a database — which is why the scoped semantic reset does not reach it (behaviour B2, preserved) |
| `text_attribute` | 4 | `:TextAttribute` | declared and referenced by `reset.py`, never written by GSF today |
| `analysis` | 4 | `:Analysis` | same |

## Zones

| Table | Cols | Replaces | Notes |
|---|---|---|---|
| `zone` | 5 | `:Zone` / `:disableZone` | the label swap becomes `enabled boolean NOT NULL DEFAULT true` |
| `zone_target` | 5 | `:ZONE_OF` | polymorphic: three nullable FKs + `CHECK (num_nonnulls(database_id, schema_id, table_id) = 1)` |

Three nullable FKs beat a `(kind, id)` pair here because they keep real
referential integrity: deleting a table removes its zone membership by cascade.
A `(kind, id)` pair cannot carry an FK at all and would leave dangling grants
behind — an access-control bug, not untidiness.

## Views

| View | Purpose |
|---|---|
| `join_edge` | the traversal surface `find_join_path` walks: `CONTAINS` and `HAS_ATTRIBUTE` in both directions, `SEMANTIC_FK` **outgoing only** |

That asymmetry is load-bearing and is why the Cypher needed
`apoc.path.expandConfig` rather than a plain variable-length pattern. Traversed
undirected, a path would hop from one FK column up to a shared target attribute
and back down a *different* FK column, inventing a join between two columns
that merely reference the same thing. Here it is simply a row that is not
emitted.

## Not yet modelled

- `IS_A`, `ROLE`, `UNION` — declared in the vocabulary but never written by
  GSF. No tables created; add them with a migration if a writer appears.
- Anything Phase 4 discovers the write path needs. The ERD is expected to move
  once real writes hit it.
