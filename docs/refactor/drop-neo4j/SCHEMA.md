# Drop Neo4j — Postgres schema reference

Living, human-readable description of the `gsf` Postgres schema. **Regenerated
whenever a migration lands**, so nobody has to read Alembic revisions in order
to know what the tables look like.

Design rationale lives in [PLAN.md § Schema design](PLAN.md#schema-design) —
this file describes what *is*, not why.

**Status: not yet created.** The schema lands in Phase 3
(`gsf/dal/pg/schema.py` + Alembic revision `0001`). Until then this file
records only the target shape.

---

## Conventions

- Namespace **`gsf`** — never `public` (Prisma's) or `vdb` (langchain_postgres').
  Set via `MetaData(schema="gsf")`, never `search_path`.
- Primary keys are `text` UUIDs, `DEFAULT gen_random_uuid()::text`.
- Catalog tier is prefixed `catalog_`; semantic tier uses the bare lowercased
  singular label.
- Table names singular, `snake_case`.
- Edge tables named `<source>_<target>` in the direction the relationship
  points, or after the relationship where that reads better.

## Planned tables

### Catalog tier

| Table | Replaces | Notes |
|---|---|---|
| `catalog_database` | `:Database` | `connection jsonb`; Vault path unchanged |
| `catalog_schema` | `:Schema` | FK → `catalog_database` |
| `catalog_table` | `:Table` | FK → `catalog_schema`; `table_type` |
| `catalog_column` | `:Column` | FK → `catalog_table`; `sample_values`, `is_unique`, `is_nullable`, `ordinal_position` |
| `column_foreign_key` | `:FOREIGN_KEY` | Column → Column |
| `table_join` | `:JOIN` | Table → Table, carries `join_columns` |
| `sql_query` | `:Sql` | identity hashed — text can exceed the btree row limit |
| `sql_query_table` | `:SQL` | Sql → Table |

`CONTAINS` is **not** an edge table — it's the parent FK column on each tier.
That is what collapses the old APOC reset cascade into `ON DELETE CASCADE`.

### Semantic tier

| Table | Replaces | Notes |
|---|---|---|
| `term` | `:Term` | `synonyms`, `name_certified`, `description_certified` |
| `column_attribute` | `:ColumnAttribute` | 5-part merge key as partial unique indexes; `table_id` is plain `text`, **no FK** (the importer legitimately writes `''`) |
| `column_attribute_link` | `:HAS_ATTRIBUTE` ∪ `:SEMANTIC_FK` | discriminated by `kind` |
| `sql_attribute` | `:SqlAttribute` | `source` ∈ manual/sql/table/bridgeTable |
| `text_attribute` | `:TextAttribute` | |
| `analysis` | `:Analysis` | |
| `custom_analysis` | `:CustomAnalysis` | |
| `pql_analysis` | `:PqlAnalysis` | never attached to a database |
| `table_term` | `:REPRESENTS` | Table → Term |

### Zones

| Table | Replaces | Notes |
|---|---|---|
| `zone` | `:Zone` / `:disableZone` | the label swap becomes `enabled boolean NOT NULL DEFAULT true` |
| `zone_target` | `:ZONE_OF` | polymorphic: three nullable FKs + `CHECK (num_nonnulls(...) = 1)` |

### Views

| View | Purpose |
|---|---|
| `join_edge` | unified edge set for `find_join_path` traversal — `SEMANTIC_FK` emitted in one direction only |

`imported_id text` (non-unique index) is carried on every entity that
`_resolve_entities_batch` touches: `catalog_database`, `catalog_schema`,
`catalog_table`, `catalog_column`, `term`, `column_attribute`, `sql_attribute`,
`custom_analysis`.
