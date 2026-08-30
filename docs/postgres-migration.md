# Neo4j → Postgres: what changed and why

GSF's catalog and semantic layer used to live in Neo4j, written through a fork
of NeMo-Retriever's ingestion code. They now live in Postgres, queried with
SQLAlchemy Core from `gsf/dal/`. This is the orientation document for anyone
reading the diff or working on the store afterwards.

The store is **native relational tables** — not Apache AGE, not a graph
emulation. Traversals that were Cypher are now either joins, a recursive CTE,
or an explicit breadth-first search. There is **no data migration**: the change
landed while GSF was greenfield, so nothing had to be carried across.

---

## 1. Bugs found in GSF

These are defects in GSF's own behaviour, found because porting a query forces
you to state what it actually returns. Each was found by *measuring* against a
real fixture, not by reading code.

### Fixed

**1. Partitioned tables and materialized views were silently missing.**
`gsf/connectors/postgres.py` filtered `relkind` to a set that excluded `'p'`
(partitioned parent) and `'m'` (materialized view). `relispartition = false`
correctly hid partition *children*, but the parent was dropped with them, so an
entire queryable table was absent from the catalog with no error anywhere.
Pagila's `payment` table is the fixture case.

**2. No column change survived a re-ingest.** The incremental diff compared
freshly parsed columns against stored ones, but the comparison never ran, so a
column whose type or nullability changed upstream kept its original values
forever. Only a full re-ingest of the database corrected it.

**3. Every column in every export claimed to be nullable.**
`catalog_column.is_nullable` stored the *strings* `information_schema` reports —
`'YES'` and `'NO'` — and the export read them with `bool(raw)`. `bool('NO')` is
`True`. Measured on the fixture, **112 of 218 columns were wrong**, and an
export is what another deployment imports as truth. Nothing failed, because a
bool is exactly what the schema expects and `True` is a plausible value for it.

This was first patched with explicit string parsing, then fixed properly: the
column is now a real `Boolean`, and each connector converts at its own edge.
See §3 for the full treatment, since it turned out to be the most instructive
bug of the set.

### Preserved deliberately, and pinned by tests

These two are wrong, or at least surprising, but changing them is a product
decision rather than a migration one. They behave exactly as they did before;
tests now assert the current behaviour so a future fix is a deliberate act.

**4. The SqlAttribute suggester never ranked.** Suggestions came back in
arbitrary order. `total_counter` is stored and is what a ranking would use —
`sql_attribute_suggester._usage_score` is where it would go.

**5. `candidates.expand_info` is asymmetric.** A `SqlAttribute` with no
statement vanishes from the result entirely; a `CustomAnalysis` with no
statement comes back with `sql: ""`. Two shapes for the same condition.

(A third entry here — a cross-database Term making an export unimportable —
was **fixed** in review rather than preserved. The export still records every
table representing a term, in scope or not, because that is honest about a term
it only partly owns; the *import* now skips references the document does not
carry, instead of aborting. Structural references, such as a foreign key's
endpoints, are still a hard error.)

---

## 2. ERD

Everything lives in the `public` schema — never `frontend` (Prisma's) and never
`vdb` (langchain_postgres'). Every foreign key below is `ON DELETE CASCADE`.

Embeddings are **not** here: they live in the `vdb` schema, in the two
langchain_postgres collections `PostgresVDB` creates and indexes at runtime. A
vector is linked to its entity by an id inside that collection's metadata, so
deleting an entity does not delete its vector — see `delete_by_database` and
`delete_all` on the store for how a stale collection is cleared.

```
                        ┌──────────────────┐
                        │ catalog_database │
                        └────────┬─────────┘
                                 │ database_id
                        ┌────────▼─────────┐
                        │  catalog_schema  │
                        └────────┬─────────┘
                                 │ schema_id
                        ┌────────▼─────────┐
                        │  catalog_table   │
                        └────────┬─────────┘
                                 │ table_id
                        ┌────────▼─────────┐
                        │  catalog_column  │
                        └──────────────────┘
```

The hierarchy is **parent FK columns**, not association rows. That is the one
structural decision everything else leans on: deleting a database is a single
statement, because schemas, tables and columns cascade with it.

Association tables (both endpoints cascade):

| Table | Links | Notes |
|---|---|---|
| `column_foreign_key` | column → column | physical FKs read from the source |
| `column_join` | column → column | `refs text[]`, accumulated from statements |
| `column_union` | column → column | `refs text[]`, same |
| `table_join` | table → table | |
| `sql_query_table` | sql_query → table | |
| `sql_query_column` | sql_query → column | |
| `column_has_attribute` | column → column_attribute | |
| `column_semantic_fk` | column → column_attribute | LLM-inferred, one direction |
| `column_attribute_term` | column_attribute → term | |
| `sql_attribute_term` | sql_attribute → term | |
| `table_term` | table → term | |
| `sql_attribute_sql` | sql_attribute → sql_query | |
| `custom_analysis_sql` | custom_analysis → sql_query | |
| `zone_target` | zone → database / schema / table | access scoping |

Standalone: `term`, `column_attribute`, `sql_attribute`, `custom_analysis`,
`pql_analysis`, `analysis`, `text_attribute`, `sql_query`, `zone`.

One view: **`join_path_edge`** — see §3.

---

## 3. Design decisions

### Alembic owns the schema; the DAL never creates it

`gsf/dal/schema.py` is the single source of truth for the tables.
`gsf/dal/session.py` owns the pooled engine. Neither ever issues DDL. All
structural change goes through Alembic:

```
uv run alembic revision --autogenerate -m "..."
uv run alembic upgrade head
```

**This could have been Prisma instead, and deliberately isn't.** The frontend
already runs Prisma against the *same Postgres database* — `schema.prisma`
declares `schemas = ["frontend"]` and owns the auth, conversation and
configuration models — and Prisma's multi-schema support would happily have
taken `gsf` as a second entry. One migration tool for the whole database is a
real option and would have meant one `migrate` job instead of two.

It is split on **separation of concerns**: the backend owns the catalog schema,
so the backend's own toolchain migrates it. `gsf/dal/schema.py` is the source
of truth a Python developer edits, autogenerate diffs against that same object,
and a backend schema change never requires touching a frontend file or running
`pnpm`. The cost is that the two tools must stay out of each other's way — which
is exactly what `include_object` below enforces, and why each migrator is
scoped to its own Postgres schema (`public` vs `frontend`) rather than sharing one.

Two pieces of `alembic/env.py` are load-bearing and easy to break:

**`include_object` filters to the `public` schema.** The same database also holds
Prisma's tables (`frontend`) and langchain_postgres' vector tables (`vdb`).
Without the filter, autogenerate sees them as untracked and proposes dropping
them — verified, not assumed: putting a table Alembic does not know about into
the filtered schema makes the next `--autogenerate` emit `op.drop_table` for it.
The corollary of GSF owning `public` is that this now cuts both ways: a table
created there by anything other than a migration will be proposed for deletion,
because from the filter's side it is indistinguishable from one removed from the
model. The filter **defaults to exclude**: Alembic passes tables, columns,
indexes and constraints through the same hook and they disagree about how to
reach their schema — some carry `.schema`, some carry a `.table` that is a
`Table`, some carry a `.table` that is only its name as a string. An object
whose schema cannot be established is treated as not ours.

**`alembic_version` lives in `gsf`**, so `env.py` runs
`CREATE SCHEMA IF NOT EXISTS gsf` before the first migration can record itself.

**Views are on a separate `MetaData`.** Alembic autogenerates by diffing
`METADATA` against the database and cannot tell a view from a table. A view
registered on `METADATA` would be emitted as `CREATE TABLE` on the next
autogenerate and then reported as permanent drift. So `gsf/dal/schema.py`
defines a second registry, `VIEWS`, holding `join_path_edge` as a `Table`
object queries can `select()` from, while the DDL string
`JOIN_PATH_EDGE_VIEW_SQL` stays the single definition. The migration executes
that string by hand — **autogenerate will never emit it**, so a squash or
regeneration that forgets it silently breaks join-path finding.

There is exactly **one baseline migration**, and it is re-squashed rather than
extended whenever migrations accumulate. Nothing is deployed with an earlier
revision applied, so there is no upgrade path to preserve, and one file that
matches `schema.py` beats a chain that has to be replayed mentally to know the
current shape. A database built by an earlier chain is structurally identical
and can be adopted with:

```
uv run alembic stamp --purge head
```

Plain `stamp head` fails on such a database — Alembic cannot reason about a
stored revision that no longer exists in the version directory.

The revision id changes on every squash, and `backend.alembicRevision` in
`helm/gsf/values.yaml` must track it: the backend's `wait-for-catalog-schema`
init container blocks until `public.alembic_version` matches that value, so a
stale pin leaves every Pod waiting forever. `gsf/dal/tests/test_migration_revision.py`
fails if the two drift, and if there is ever more than one head.

### Join paths

This is the traversal that survived least intact, and the one worth
understanding before changing anything near it.

In Neo4j a join path was `apoc.path.expandConfig` over a labelled graph. In
Postgres the edges are spread across `catalog_column.table_id`,
`column_has_attribute` and `column_semantic_fk` — three different tables with
different shapes. The view **`join_path_edge`** normalises them into one
`(src_kind, src_id, dst_kind, dst_id)` relation, so the search has a single
thing to walk:

```
column           → table            (via catalog_column.table_id)
table            → column           (the reverse)
column           ↔ column_attribute (column_has_attribute, both directions)
column           → column_attribute (column_semantic_fk, one direction only)
```

`SEMANTIC_FK` is deliberately emitted **one way** in the view while being read
both ways elsewhere — see the comment in `gsf/dal/schema.py`.

`find_join_path` in `gsf/dal/attributes.py` is a **level-at-a-time BFS with a
shared visited set**, not the recursive CTE that is the obvious translation.
The distinction matters: a recursive CTE carries a *per-path* visited set and
will re-expand nodes already reached by a shorter route, which on this graph
explodes. BFS with one shared set returns the shortest path and visits each
node once. Within a level, the first edge into a node wins — without that
guard, a parent pointer can be overwritten by a longer route discovered in the
same batch, which is no longer BFS.

**`MAX_PATH_DEPTH = 30` is not arbitrary.** A "hop" in product terms is four
edges — `Column → ColumnAttribute → Column → Table → Column` — so an *n*-hop
path is `4n - 2` edges. An ordinary 4-hop path is 14. A ceiling of 10 looks
generous and would have silently returned "no path" for it. The value is a
depth in *edges*, and the two units are easy to confuse.

### `is_nullable`, and why it became a Boolean

The column now stores a real `Boolean`. NULL means "the connector could not
determine it", which callers read as nullable. Each connector converts at its
own edge, because the conversion differs per engine:

| Connector | Conversion | Source shape |
|---|---|---|
| Postgres | `NOT a.attnotnull` | already boolean in `pg_attribute` |
| SQLite | `not notnull` | `PRAGMA table_info` flag |
| HeavyDB | `bool(type_info.nullable)` | driver returns a bool |
| DuckDB | `is_nullable = 'YES'` | `information_schema` text |
| Snowflake | `IS_NULLABLE = 'YES'` | `TEXT`, documented `'YES'`/`'NO'` |
| Databricks | `is_nullable = 'YES'` | `STRING`, documented `'YES'`/`'NO'` |

`gsf/catalog/normalize.py:coerce_nullable` is a safety net at the ingest
boundary for third-party connectors that still return the SQL-standard text. It
**raises on anything it does not recognise** rather than defaulting — silently
guessing is what made the original bug invisible.

Note the bug had a second home: `update_column_props_by_arguments` guarded with
`if is_nullable:`, which is harmless for the truthy string `'NO'` but drops
every `False` once the value is a real boolean, storing NULL and reading back as
nullable. The string fix would have survived its own repair.

### Other decisions worth knowing

- **`CONTAINS` is a parent FK column, not an association row.** This is what
  makes `reset.py`'s delete a single statement. It also means writing that one
  relationship is an `UPDATE` of the child while every other link is an
  `INSERT` — an asymmetry paid in exactly one place.
- **`refs` arrays accumulate, then de-duplicate.** Each statement that joins two
  columns adds its own reference, so the link records how often and where the
  join was observed. The graph appended without de-duplicating and grew the
  array on every re-ingest of the same statement; that was a leak, not a
  behaviour worth reproducing.
- **Per-month `count_{month}_{year}` counters are not written.** No reader could
  ever parse those names. `total_counter` is kept and is what a fix would use.
- **The `deleted` filter is dropped.** Nothing in GSF ever wrote `deleted`, so
  the guard was always true.
- **Vector-store `label` metadata is not a Neo4j leftover.** One embedding
  collection holds Tables, Columns, Sql and CustomAnalysis; `label` is how a
  search restricts to one kind, and the strings are *persisted* in
  `langchain_metadata`. Renaming them means re-embedding everything.

---

## 4. How this was tested

The rule throughout: **a test that only proves the code runs proves nothing
about a port.** Fidelity had to be measured against recorded behaviour or
against the source engines themselves.

### Golden replay — the primary oracle

122 DAL reads were captured against the **Neo4j** implementation before it was
removed, normalised (ids → stable tokens, timestamps redacted, undirected edges
oriented consistently) and committed. The Postgres implementation must return
identical output. The golden suite is 128 tests over those 122 recordings.

Re-recording is a last resort. A golden that changes because the code changed
is the test doing its job:

```
uv run --no-sync python -m dev_tools.capture_dal_golden   # only when intended
```

List *order* is deliberately not covered — most DAL queries lack a total
`ORDER BY`.

### Ground truth against live engines

For `is_nullable`, correctness was checked against the source databases rather
than against GSF's own opinion: ingest Chinook (SQLite) and Pagila (Postgres)
into a throwaway catalog, then compare all **209 columns** with
`PRAGMA table_info` and `pg_attribute` respectively. **209/209 match.** The
distribution is 110 non-nullable / 99 nullable — the bug's signature was every
column agreeing.

### Byte-level equivalence harness

For the ingest-path refactor, the test suite was not enough: it can pass while
the *written rows* drift. So the ingest was fingerprinted — every catalog table
after a from-scratch ingest, UUIDs replaced by natural-key tokens, timestamps
redacted, arrays sorted. The harness was validated first: two independent
ingests must produce identical output.

**This caught a bad proof.** The first fingerprint compared identical — but
`sql_query`, `sql_query_table`, `sql_query_column`, `column_join` and
`column_union` were all *zero rows*, because the fixtures generate no SQL
history. The run exercised none of the changed code. A second scenario was
built to drive statements through `parse_queries_df` → `add_query` (SQL links,
two joins, a union, deliberate repeats for `refs` accumulation), then run
against the pre-refactor code via `git stash` and diffed. Both paths: identical.

The same technique proved the squashed migration: build one database through
the old four-migration chain and one through the new baseline, then diff full
structure — 128 columns, 145 constraints, 66 indexes, 2 checks, 1 view. The
only differences were Postgres-generated `NOT NULL` constraint names, which
embed the table OID and differ between any two databases.

### Fixtures

`testdb.sql` (16 lines) was replaced with **Pagila** and **Chinook**. Pagila is
the only fixture with a partitioned table and a materialized view — the two
relation kinds bug #1 dropped. Chinook provides a self-referential FK
(`Employee.ReportsTo`) for join-path testing.

### Surface freezing

`gsf/dal/tests/dal_surface.json` snapshots every public DAL signature. A renamed
keyword or dropped argument breaks callers silently; this fails instead.

### Live end-to-end

Run with real NVIDIA models: 246 embeddings, LLM semantic compilation
(10 terms, 53 attributes, 11 SEMANTIC_FKs), `find_join_path` over real data
including `Employee.ReportsTo`, and NL-to-SQL returning correct answers
(3503 tracks; Iron Maiden 21 / Led Zeppelin 14 / Deep Purple 11).

`POST /api/chat/completions` **cannot be tested through `TestClient`** — the
worker pool uses `mp.get_context("spawn")` and needs a real server. Run uvicorn.

### Current state

`713 tests` — 712 passed, 1 skipped. `ruff check` clean.

---

## 5. Migration notes and gotchas

Things that cost time, recorded so they cost it once.

**A test for an unscoped destructive function has no fixture boundary by
construction.** A `delete_all_data(None)` in a reset test wiped the shared
fixture database. The fix is a `_rolled_back()` transaction helper. Treat any
test of an unscoped delete as needing an explicit boundary.

**Catch-and-degrade turns hard errors into plausible empty results.** Two real
bugs — a `SELECT DISTINCT` with an `ORDER BY` key not in the select list, and a
`::text` cast colliding with bind-parameter parsing — were masked by
`except: return []`. Both looked like "no results" rather than failures.

**`::` in `text()` collides with bind parameters.** Use `CAST(x AS text)`.

**A UNION wrapped in `.subquery()` silently stops correlating.** `_terms_count`
reported the same count for every table. Fixed with an explicit
`.correlate(owner, s.term)`. The test asserts a table with *no* terms alongside
one with two — a test using only non-empty cases would have passed.

**Postgres collation is not Python's `sorted()`.** A test comparing
`ORDER BY name` against Python sorting fails on `order_tag` vs `orders`.
Compare against the database's own ordering.

**Tests that create named entities must scope their names.** `test_users.py`
created zones outside its own prefix; 32 accumulated and broke a golden.
Similarly, tests that error during *fixture setup* never reach their cleanup —
95 leftover databases accumulated that way and broke five goldens for reasons
that had nothing to do with the code under test.

**`nemo-retriever` is pinned to `==26.8rc1`.** The `tabular` extra was the last
thing pulling the Neo4j driver into the tree, so `SQLDatabase` and the
embedding-row builder were vendored into `gsf/connectors/base.py` and
`gsf/utils/embedding_rows.py`. The latest *stable* release (26.5.0) is far too
old — most of the API GSF imports does not exist in it. `langgraph` became a
direct dependency because it used to arrive through that extra.
