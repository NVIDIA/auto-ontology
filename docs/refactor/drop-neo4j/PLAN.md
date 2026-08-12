# Drop Neo4j: migrate the graph to Postgres

## Context

GSF stores its data catalog and semantic layer in Neo4j, but the model is not
actually graph-shaped: the spine is a strict hierarchy
(`Database → Schema → Table → Column`) with a semantic overlay (`Term`,
`ColumnAttribute`, `SqlAttribute`, `CustomAnalysis`, `Sql`, `Zone`). Only three
call sites do anything a relational schema handles badly.

Running Neo4j alongside Postgres costs real things today:

- **No cross-store joins.** `expand_info` (`gsf/dal/candidates.py`) enriches
  pgvector hits with node properties as a second round trip; zone scoping
  (`gsf/dal/users.py`) reconciles Postgres roles against a Neo4j graph in
  application code.
- **No referential integrity or types.** Everything is a property bag.
- **Two databases to operate** — a Neo4j StatefulSet, 4 PVCs, an APOC plugin,
  and a `wait-for-neo4j` init container, for data that Postgres already
  neighbours.
- **The catalog write path isn't even in this repo.** It lives in
  NeMo-Retriever (`nemo_retriever/tabular_data/ingestion/`), so the schema GSF
  reads is written by a third-party library GSF cannot change.

**Outcome:** one datastore. Postgres holds the catalog, the semantic layer, the
vectors (pgvector), and the app tables. Neo4j, APOC, and the `neo4j` driver are
deleted, and the catalog write path becomes GSF-owned code under
`gsf/catalog/`.

### Decisions already made

| Decision | Choice |
|---|---|
| Storage model | Native relational tables (not Apache AGE, not node/edge JSONB) |
| Query layer | SQLAlchemy Core — no ORM (ORM can be layered on the same metadata later) |
| Migrations | Alembic, Python-side; Prisma keeps owning `public` |
| Existing Neo4j data | **Greenfield** — dropped, re-ingested from source |
| Cutover | Swap the bodies of `gsf/dal/*.py`, keeping every signature identical |

---

## Bugs this refactor has surfaced

Not sought out — every one turned up while measuring what the system does in
order to reproduce it. They are listed here because they are the strongest
argument for the approach, and because each one is a thing the product was
getting wrong in production.

| # | Bug | Found by | Status |
|---|---|---|---|
| 1 | **Partitioned tables and materialized views were missing from every Postgres catalog.** `relkind` excluded `'p'`, so a partitioned parent appeared nowhere; and `information_schema` lists no matviews, making `TableTypes.MATERIALIZED_VIEW` unreachable dead code | ingesting the Pagila fixture | fixed, **B3** ([003](DECISIONS.md)) |
| 2 | **No column change ever survived a re-ingest.** The column diff merged on keys neither frame had, raising `KeyError` every time — and `executor.map`'s result was never consumed, so the exception was discarded in silence. Added columns never appeared; dropped columns left ghosts the SQL generator kept querying | writing the incremental-diff tests | fixed, **B6** ([006](DECISIONS.md)) |
| 3 | **Every column exports as nullable.** `model_interchange` reads `is_nullable` with `bool(...)`, but the graph stores the strings `'YES'`/`'NO'` — and `bool('NO')` is `True` | measuring stored property types | **open**, for Phase 10 |
| 4 | **The SqlAttribute suggester never ranked anything.** Its scorer matched `count_monthly_YYYY_MM` while the writer produced `count_{month}_{year}`, so every expression scored 0.0 and ordering was dict insertion order | deciding how to store monthly counters | **open** — documented in place, behaviour preserved ([007](DECISIONS.md)) |

Three of the four are silent: no error, no log, no failing test. That is the
pattern worth noting — a schemaless store lets a name mismatch sit undetected
indefinitely, because nothing declares what a name is supposed to be. Two of
them (2 and 4) were a regex or a merge key disagreeing with a writer that no
schema constrained.

**Bugs 3 and 4 are still open, deliberately.** Both change product output —
what a model export claims about nullability, and which SqlAttributes get
suggested — so both belong in their own change with their own tests rather than
inside a port. Bug 3 is Phase 10's, which must fix it and un-skip the two
`test_export_model_*` tests together. Bug 4 has its fix written out in
`_usage_score`'s docstring, ready to apply.

The line between fixing and preserving is whether the bug destroys something.
Bugs 1 and 2 lost catalog entries, so reproducing them faithfully would have
meant porting data loss. Bugs 3 and 4 produce wrong-but-harmless output, and
preserving them keeps the port honest about what it changed.

---

## This plan lives in the repo

This document is the source of truth for the refactor. It was drafted outside
the repo and landed here in the first commit of Phase 0; **this repo copy is
authoritative** and any copy under `~/.claude/plans/` is dead — do not edit it.

```
docs/refactor/drop-neo4j/
  PLAN.md        ← this document; the source of truth. Amended in place, never replaced.
  PROGRESS.md    ← append-only log of everything that lands
  DECISIONS.md   ← numbered decision records for anything not settled here
  SCHEMA.md      ← living table/ERD reference, updated as tables land
```

### Documenting autonomous work

This refactor spans 8–11 weeks and much of it will be executed unattended. The
rule is that **the documentation is part of the change, not a follow-up**:

1. **No implementation commit without a same-commit doc update.** A commit that
   ports DAL functions also appends to `PROGRESS.md`. If they're separable, the
   change was too big.
2. **`PROGRESS.md` entries are append-only** and carry: date, phase, what
   landed, files touched, tests added, Done-criteria met or explicitly not met,
   and what's next. Never rewrite history in it — a wrong earlier entry gets a
   correcting entry, not an edit.
3. **Any deviation from `PLAN.md` requires a `DECISIONS.md` record *before* the
   code lands.** Format: context / decision / consequences / what it supersedes.
   This covers both "the plan said X but X doesn't work" and anything the plan
   simply didn't anticipate. Then amend `PLAN.md` in the same commit so the two
   never disagree.
4. **Behaviour changes get flagged explicitly**, in `DECISIONS.md` and in the PR
   description. Two are already known: the narrower reset deletes (§Schema
   design) and `PqlAnalysis` scoping. Any further ones found mid-port must be
   surfaced rather than absorbed silently — this is the single highest risk of
   unattended execution.
5. **`SCHEMA.md` is regenerated whenever a migration lands**, so there is always
   a current human-readable description of the tables that doesn't require
   reading Alembic revisions in order.
6. **Every PR states its phase and its Done criteria verbatim**, with each one
   checked or explicitly waived. A waived criterion needs a `DECISIONS.md`
   entry.
7. **Open questions go in `PROGRESS.md` under a "Blocked / needs a human"
   heading** rather than being resolved by guess. Anything touching zone
   scoping or access control belongs there by default.

Git conventions follow `CLAUDE.md`: branch from `main`, concise imperative
commit messages, no `Co-Authored-By` trailers.

### Merge from `origin/main` before every phase

**Start each phase by fetching and merging `origin/main`**, before writing any
code:

```bash
git fetch origin && git merge origin/main
```

This refactor runs 8–11 weeks against an actively developed `main`, and it
touches `gsf/dal/` — which nearly every feature branch also touches. Deferring
the merge means conflict-resolving a phase's worth of ported SQL against
someone else's Cypher edits, in a file whose two versions no longer resemble
each other. Merging at a phase boundary keeps each conflict small and puts it
at the one moment when nothing is half-ported.

Record the merge in `PROGRESS.md` as the phase's first entry: the
`origin/main` SHA merged, and whether it conflicted. If a merge brings in new
Cypher in a module already ported, that's a `DECISIONS.md` entry — the new
behaviour has to be carried into the Postgres implementation, not silently
dropped by taking "ours".

---

## Scope

**~5,000 lines of Cypher** across 16 modules in `gsf/dal/`:
`model_interchange.py` (1550), `terms.py` (1355), `datasources.py` (885),
`exploration.py` (764), `sql_attributes.py` (649), `attributes.py` (433),
`custom_analyses.py` (405), `zones.py` (356), `pql_analyses.py` (287),
`reset.py` (215), `users.py` (178), `candidates.py` (170),
`cypher_fragments.py` (95), `neo4j_tx.py` (85), `connections.py` (83).

**Plus ~2,500 lines forked out of NeMo-Retriever.** The fork is bigger than
"the ingest path" — verified by grep, the library's Neo4j-coupled code is on
the *request* hot path too:

- `add_query` (`ingestion/dal/queries_dal.py`) — 3 sites incl.
  `gsf/server/sql_attributes/service.py`, `gsf/server/custom_analyses/service.py`
- `parse_query_single` (`ingestion/services/queries.py`) — 2 sites incl.
  `gsf/retrieval/text_to_sql/agents/sql_parse_validation.py`
- the 499-line `Schema` model — consumed by
  `gsf/retrieval/data_access/graph_schemas.py`, i.e. every SQL validation
- `Neo4jNode` — 4 sites; `reserved_words` (`Labels`/`Edges`/`Props`/
  `TableTypes`) — ~29 import statements across 23 modules

**Stays on the library, untouched:** `SQLDatabase` (16 sites — all of
`gsf/connectors/`), `Retriever`, `VDB`, `EmbedParams`, `rerank_hits`,
`IngestVdbOperator`, `_BatchEmbedActor`, and
`TabularFetchEmbeddingsOp` (verified Neo4j-free — its only `neo4j` reference is
the literal `f"neo4j:{node_id}"` path string).

---

## Schema design

New Postgres schema **`gsf`** — not `public` (Prisma's, and `prisma db push`
drift-reconciles it) and not `vdb` (langchain_postgres'). Set it on the
metadata, never via `search_path`:

```python
METADATA = MetaData(schema="gsf")
```

`search_path` is per-session, and the DAL runs on pooled connections from
FastAPI handlers, a spawned chat-worker subprocess, and ingestion threads — a
stray `SET search_path` would be a silent cross-schema hazard.

**Keep UUID primary keys, as `text`, defaulting to `gen_random_uuid()::text`.**
Ids are consumed as opaque strings in three places that would otherwise need to
learn a second key: the pgvector row key (`f"neo4j:{node_id}"` in
`gsf/utils/embedding.py:76`), the frontend's pipe-separated focus param
(`frontend/lib/data/data-catalog-path.ts`), and `_resolve_entities_batch`'s
`imported_id OR id` match. `text` over `uuid` because every DAL signature is
`id: str`, and a malformed id from a URL should yield an empty result — not a
`22P02` 500.

**Naming:** catalog tier gets a `catalog_` prefix (`catalog_database`,
`catalog_schema`, `catalog_table`, `catalog_column`) — one rule that sidesteps
`table`/`column` being reserved and `schema` colliding with
`information_schema`. Semantic tier keeps the lowercased singular label:
`term`, `column_attribute`, `sql_attribute`, `custom_analysis`, `pql_analysis`,
`zone`. `Sql` → `sql_query`. Edge tables: `column_foreign_key`, `table_join`,
`sql_query_table`, `table_term` (REPRESENTS),
`column_attribute_link` (HAS_ATTRIBUTE ∪ SEMANTIC_FK, discriminated),
`zone_target`.

**Key modelling calls:**

- **`CONTAINS` becomes a parent FK column, not an edge table.** This is what
  collapses `reset.py`'s APOC cascade into `ON DELETE CASCADE`.
- **`ZONE_OF` polymorphism** (targets `Database|Schema|Table`) →
  one `zone_target` table with three nullable FKs and
  `CHECK (num_nonnulls(database_id, schema_id, table_id) = 1)`.
- **The `Zone`/`disableZone` label swap** → `zone.enabled boolean NOT NULL
  DEFAULT true`. Confirmed nothing depends on label semantics beyond the swap.
- **`ColumnAttribute`'s 5-part merge key** → partial unique indexes (not a
  composite PK — NULL handling), and `table_id` stays plain `text NOT NULL`
  with no FK, because the importer legitimately writes `''`
  (`model_interchange.py:1260`).
- **`Database.connection`** → `catalog_database.connection jsonb`. Vault path
  unchanged.
- Keep `imported_id text` + non-unique index on every entity
  `_resolve_entities_batch` touches.

**Hard query translations** (designed, with SQL drafted):

| Today | Becomes |
|---|---|
| `reset.py` — `apoc.periodic.iterate` + `apoc.path.subgraphNodes` | `ON DELETE CASCADE` + a few explicit deletes |
| `exploration.py` — 1–2 hop expansion, `MAX_EXPLORATION_GRAPH_NODES` | plain joins; ~7 genuinely novel queries |
| `attributes.py:335` `find_join_path` — `apoc.path.expandConfig`, mixed directionality, `maxLevel: 30` | a `join_edge` view + BFS — **own spike, designed below** |
| `cypher_fragments.py` `coalesce(..., head([...]))` | `COALESCE`/`NULLIF` + LATERAL, in a new `gsf/dal/sql_fragments.py` |
| `model_interchange.py` batched `UNWIND` + `MERGE` | `insert().on_conflict_do_update()` + executemany, one generic helper in `gsf/dal/bulk.py` |

Two deliberate behaviour changes to call out in PR descriptions: the explicit
deletes are *narrower* than `subgraphNodes`, which today bleeds across
databases via shared `Term`/`Sql` nodes; and `_delete_semantic_nodes`'s scoped
branch does **not** collect `PqlAnalysis` today — reproduce that, don't
silently fix it.

### `find_join_path` — the one query without a mechanical translation

`labelFilter: '-Schema'` (`attributes.py:363`) does more work than it appears.
`CONTAINS` is traversed undirected, so from a `Table` you could walk *up* to
`Schema` and out to every other table in the database; blacklisting `Schema`
severs that. The reachable graph is therefore only three edge kinds:

| Edge | Between | Direction |
|---|---|---|
| `CONTAINS` | Table ↔ Column | undirected |
| `HAS_ATTRIBUTE` | Column ↔ ColumnAttribute | undirected |
| `SEMANTIC_FK` | Column → ColumnAttribute | **outgoing only** |

The `SEMANTIC_FK` directionality is load-bearing and must survive the port
exactly: undirected, you hop from one FK column up to a shared target attribute
and back down a *different* FK column, fabricating a join between two columns
that merely reference the same thing (two `person_id` columns joined to each
other). `Database` is unreachable, so a cross-database path can only arise via
a **shared `ColumnAttribute`** — which is precisely why the rejection at `:406`
exists.

**Step 1 — a unified edge view.** The asymmetry that needed a special APOC
config becomes "don't emit the reverse row":

```sql
CREATE VIEW gsf.join_edge AS
SELECT 'column'::text, c.id, 'table'::text, c.table_id FROM gsf.catalog_column c
UNION ALL SELECT 'table', c.table_id, 'column', c.id FROM gsf.catalog_column c
UNION ALL SELECT 'column', l.column_id, 'column_attribute', l.attribute_id
    FROM gsf.column_attribute_link l WHERE l.kind = 'HAS_ATTRIBUTE'
UNION ALL SELECT 'column_attribute', l.attribute_id, 'column', l.column_id
    FROM gsf.column_attribute_link l WHERE l.kind = 'HAS_ATTRIBUTE'
UNION ALL SELECT 'column', l.column_id, 'column_attribute', l.attribute_id
    FROM gsf.column_attribute_link l WHERE l.kind = 'SEMANTIC_FK';  -- no reverse
```

Needs indexes on `catalog_column(table_id)` and
`column_attribute_link(column_id, kind)` / `(attribute_id, kind)`.

**Step 2 — the trap.** A recursive CTE with a `path_ids` visited array is the
obvious translation and is *correct*, but it is **not** what APOC did.
`uniqueness: 'NODE_GLOBAL'` is BFS with a shared visited set — linear in the
reachable component. A recursive CTE's visited array is **per-path**, so every
distinct route to a node is expanded separately: exponential in a dense graph.
Unnoticeable on Pagila (~120 columns); severe on a 5,000-column customer
database where hub attributes like `customer id` create massive fan-out. And
`ORDER BY depth ... LIMIT 1` forces full materialisation, so it cannot
short-circuit either.

**Step 3 — build both.** Write the recursive CTE first as the reference
implementation (~20 lines, easy to verify against the Neo4j goldens), then make
production use **level-at-a-time BFS driven from Python**, one indexed query
per level with the visited set held in Python — which reproduces `NODE_GLOBAL`
exactly. Real join paths are 2–4 hops, so that's typically 3–5 trivial queries:
more round trips, but work bounded by component size rather than path count. It
also makes the depth bound visible and testable (cap at 10; APOC's `maxLevel:
30` is far beyond any real path) and the visited set directly assertable.

Keep the CTE as a test oracle. If a benchmark on a wide synthetic catalog shows
it holds up, delete the BFS and keep the simpler code — but decide that with a
measurement, not upfront.

Everything downstream is unchanged: the `col_nodes` filter, the pairwise
`(i, i+1)` hop construction, `fetch_col_table_contexts`, and the
cross-database rejection are already pure Python over a node list. The hop
pairing is subtle and should not be touched.

---

## Sequencing

Every phase is independently mergeable and leaves main green, because
`GSF_STORE` defaults to `neo4j` until Phase 11.

### Phase 0 — Land the docs, sever `reserved_words`, freeze the DAL surface *(2–3d)*
**First commit:** move this plan to `docs/refactor/drop-neo4j/PLAN.md`, create
`PROGRESS.md` / `DECISIONS.md` / `SCHEMA.md` stubs, add the `README.md` pointer
and the `CLAUDE.md` line. Everything after this follows the documentation rules
above.

Create `gsf/catalog/constants.py` with GSF-owned `Labels`/`Edges`/`Props`/
`TableTypes` (verbatim values, ~50 lines of pure constants); repoint all ~29
import statements. No shim — a re-export keeps a live import edge to the
package being dropped.

Add `gsf.dal.close_store()` and call it from `gsf/server/__main__.py:58`
instead of reaching into `neo4j_connection._conn`, so shutdown goes through a
public hook that Phase 3 can repoint at the SQLAlchemy engine.

> **Correction (2026-08-11).** An earlier draft of this plan claimed this also
> drops the heavy `nemo_retriever` import from the API entrypoint. It does not:
> `nemo_retriever.tabular_data.neo4j` still loads transitively through the ten
> `get_neo4j_conn` imports in `gsf/dal/*.py`. Only the *direct* import at
> `__main__.py:13` goes away. The entrypoint stops depending on driver
> internals; the import weight is unchanged until Phase 11.

Add `gsf/dal/tests/test_dal_surface.py` — snapshot every public name +
`inspect.signature` per module. This is the machine-checkable form of the
"keep every signature identical" decision.

**Done:** `docs/refactor/drop-neo4j/` exists and is linked from `README.md` and
`CLAUDE.md`; `grep -rn reserved_words gsf` empty; entrypoint no longer imports
`nemo_retriever.tabular_data.neo4j`; surface test green.

### Phase 1 — Fork the write path, still writing to Neo4j *(5–8d)*
Verbatim fork first, rewrite in Phase 4. New `gsf/catalog/`:

```
constants.py  extract.py  normalize.py  sql_parse.py  write.py  ingest.py
model/{node,schema,query}.py      # Neo4jNode → CatalogNode
parsers/{sqlglot_extractor,query_comparator,schemas_parser}.py
services/schema.py
store/{db,schemas,queries,edges,indexes}.py   ← the ONLY subtree Phase 4 rewrites
```

Everything above `store/` is storage-agnostic and forks once, permanently.
Public surface is deliberately four entry points: `ingest_catalog(connector)`,
`parse_query_single(...)`, `store.queries.add_query(...)`, and
`model.Schema`/`CatalogNode`.

**Delete `TabularSchemaExtractOp` rather than moving it** — it's a 70-line
wrapper around two function calls, and keeping it forces the fork to implement
a library ABC. Restructure `gsf/ingestion_service/ingest.py:51` from the
`Graph()` chain into straight-line calls. Wrap the private `_BatchEmbedActor`
import in a single `gsf/utils/embedding.batch_embed()` so the private-symbol
dependency lives in one line in one file.

> **Amendment (2026-08-11), [DECISION-004](DECISIONS.md).** "Verbatim" holds for
> 16 of the 17 forked modules — pinned AST-for-AST by
> `gsf/catalog/tests/test_fork_parity.py`. `extract.py` is the exception: it
> takes the **connector** rather than a library `TabularExtractParams`, and
> `store_relational_db_in_neo4j` is deleted rather than moved (a two-line
> forwarder that `ingest_catalog` now calls through). Two behaviour changes fell
> out: **B4**, `_shared_connection` no longer spans the embed step, and **B5**,
> `SemanticEmbedder` builds its embed graph per call. `store/connection.py`
> keeps the name `get_neo4j_conn` until Phase 11.

**Done:** a full `/ingest` of Pagila **and** Chinook through the forked path
produces an identical Neo4j graph to the pre-fork path — node counts per label,
relationship counts per type, sorted property dump. **This is the last moment
that comparison is possible; capture the dump as a golden.** (Fixture setup is
Phase 2, which runs in parallel — sequence the seed script first.)

### Phase 2 — Real fixture databases + golden capture *(4–5d, parallel)*
Greenfield removed the ability to diff production data, so goldens are the only
remaining fidelity oracle and the acceptance criterion for Phases 5–10. They
are only as good as the fixture, and today's `dev_tools/sql/testdb.sql` is
**16 lines — 2 tables, 1 FK, no views**. It cannot validate any of this work.

**Replace it with two real sample databases** (see
[Fixture databases](#fixture-databases) below): **Pagila** on Postgres and
**Chinook** on SQLite. Two, not one, because a single database cannot exercise
multi-database behaviour at all — and cross-database rejection
(`attributes.py:404`), `delete_by_database` scoping, and cross-database zone
scoping are exactly the things a silent regression would hide.

Then, on top of them, seed the semantic layer via `/semantic/compile` plus a
scripted set of hand-authored artifacts (terms with synonyms and certification
flags, column attributes including a `SEMANTIC_FK`, SQL attributes across all
four `source` values, a custom analysis, a PQL analysis, and 3 zones — one
disabled, one spanning both databases).

Snapshot every `fetch_*`/`get_*`/`list_*` result to `gsf/dal/tests/golden/`,
under `zone_ids=None`, a scoped list, and an empty list — zone scoping is where
a silent access-control regression would hide.

### Phase 3 — Postgres foundations *(4–6d, parallel)*
- `gsf/dal/pg/session.py` — pooled SQLAlchemy `Engine` off
  `gsf/infra/postgres.get_postgres_connection_string()`, plus `conn()` and
  `write_transaction()` mirroring `gsf/dal/neo4j_tx.py`'s contract exactly
  (ContextVar-held active connection, nesting reuses the outermost, commit on
  clean exit, rollback on any exception) so `with write_transaction():` call
  sites don't change. `gsf/connectors/postgres.py` is the pool template.
- `gsf/dal/pg/schema.py` — the full `MetaData(schema="gsf")`.
- `alembic/` wired to that metadata, `version_table_schema="gsf"`,
  `include_object` filtered to `gsf` so autogenerate never proposes dropping
  Prisma's or pgvector's tables. Migration `0001` opens with
  `CREATE SCHEMA IF NOT EXISTS gsf`. Add a `make migrate` target, a
  docker-compose service, and a Helm pre-install job modelled on
  `helm/gsf/templates/frontend-migrate-job.yaml`.
- New deps: `sqlalchemy>=2.0`, `alembic` (via `uv add`).

**Done:** `alembic upgrade head` on a blank DB creates the schema;
`alembic check` reports no drift; `prisma db push` still succeeds afterwards.

> **Gate: ERD review at the end of Phase 3 — the highest-leverage checkpoint in
> the plan.** Three shapes (polymorphic `zone_target`, the edge-property-carrying
> `sql_query_table`, and the `find_join_path` edge view) determine whether
> Phases 5–10 are mechanical or a rewrite. Review with everyone before anyone
> writes a `select()`.

### Phase 4 — Catalog writes on Postgres — **the gate** *(5–8d)*
Rewrite only `gsf/catalog/store/*.py`. Preserve `write.py`'s incremental
diffing (existing-DB detection, schema add/update/delete, `pulled` timestamp);
`apoc.merge.node.eager` → `INSERT ... ON CONFLICT DO UPDATE`; `indexes.py`
deletes entirely (Alembic owns indexes).

Nothing is readable until something writes — this blocks 5–10 hard.

**Done:** `/ingest` of both fixture databases populates `gsf.*`; row counts per
table match the Phase-2 node counts per label; views and the materialized view
land with the right `table_type`; re-ingest is idempotent; an added/renamed/
dropped table and column are diffed correctly; `/ingest/delete` cleans up one
database via cascade without touching the other.

### Phase 5 — `users` + `zones` *(2–3d)*
First read phase despite being small: `resolve_accessible_catalog_ids` /
`resolve_table_filter` are *parameters* to nearly every other read, and the
contract changes shape — `resolve_table_filter` returns a Cypher `WHERE` string
+ params today, and becomes a SQLAlchemy `ColumnElement` (or `None`).
**Decide once here** or every later phase re-litigates it. The four round trips
in `get_accessible_catalog_ids_for_zones` collapse to one query, and the
Postgres role check (`gsf/server/users/postgres_dal.py`) can now join
in-database.

### Phase 6 — `datasources` *(4–6d)*
Largest blast radius after `model_interchange`: `TABLE_COUNTS_SUBQUERY`, and
`fetch_schemas_by_ids` → `graph_schemas.get_schemas_by_ids` → `Schema` →
`parse_query_single`. Get this wrong and SQL validation silently stops
resolving tables. **Done** additionally requires `validate_sql` on a known-good
query to resolve against the PG-built `Schema` map.

**Scope: 19 of 26 functions** — the catalog-tier ones. The seven that reach
`Term`/`ColumnAttribute` moved to Phase 7 by
[DECISION-009](DECISIONS.md), because nothing writes semantic data until then
and against no data a wrong join is indistinguishable from a correct one.
`fetch_schemas_by_ids` is catalog-only, so the `validate_sql` criterion above
stays here.

### Phase 7 — `terms` + `attributes` + `sql_attributes` *(8–12d, 2 people)*
The semantic core, 2,437 lines, **plus seven functions inherited from Phase 6**
([DECISION-009](DECISIONS.md)): `fetch_tables_for_schema`,
`fetch_all_tables_without_term`, `fetch_columns_for_table`,
`fetch_tables_and_columns_by_node_ids`, `fetch_bridge_table_candidates`,
`fetch_tables_by_ids` and `fetch_table_context`.

Five of the seven need only the description fallback
(`column_description_expr` / `table_description_expr` → `sql_fragments`), so
port that first and most of the group follows. Re-check the list with
`uv run --no-sync python -m dev_tools.classify_dal_dependencies` rather than
by eye — three hand-analyses got it wrong.
They are small; they waited for the semantic fixture, which this phase creates.

Splits cleanly: one person on `terms` (+
`cypher_fragments` → `sql_fragments`), one on `attributes` + `sql_attributes`.
Carve `find_join_path` out as its own **2–3d spike** — design is in
[`find_join_path`](#find_join_path--the-one-query-without-a-mechanical-translation)
above; it ships two implementations (recursive CTE as oracle, Python-driven BFS
as production) plus its own test suite.

**Done:** goldens pass; `POST /semantic/compile` runs end-to-end on PG and
produces the same terms/attributes as the Phase-2 fixture.

### Phase 8 — `custom_analyses`, `pql_analyses`, `candidates`, `connections`, `reset` *(4–6d, 2–3 people)*
Small and mutually independent. `reset.py` *shrinks*. Good moment to retire the
`add_query(edges)` tuple protocol for `persist_sql(query_obj, owner)`.

### Phase 9 — `exploration` *(4–6d)*
Deliberately near-last: most of its 764 lines compose Phase 5/6/7 helpers, so
only ~7 queries are genuinely its own. Done earlier it's the hardest module;
done here it's mostly free. **Done** includes the degree-count invariant the
docstrings assert (a table's `relationship_count` from
`fetch_data_exploration_graph` equals its `total` from
`fetch_exploration_related_nodes`).

### Phase 10 — `model_interchange` *(6–9d)*
1550 lines but conceptually easiest of the big ones — CRUD over entities that
all exist by Phase 9. Bank the simplifications rather than porting the
workarounds: `_ensure_import_indexes` (`:478`) disappears, and the split
transaction at `:530-540` (SQL attributes applied outside the import
transaction because the library opened its own auto-commit session) collapses
into one atomic transaction. Update those docstrings or they mislead forever.

**Done:** export→import round-trip is idempotent (second import: all skipped,
zero created); `replace=True` scoped delete verified.

### Phase 11 — Flip and delete *(2–3d)*

**Prerequisite, added during Phase 7: make the golden harness backend-aware.**
This file calls `test_golden.py` the fidelity oracle and says Phases 5-10 are
graded by it. Measured under `GSF_STORE=postgres`, it grades nothing — 49 of 128
fail, including functions Phase 6 landed and verified separately. Two causes,
both in the harness rather than the DAL:

* `capture_dal_golden._fixture_ids()` resolves every fixture entity with
  hardcoded Cypher, so under Postgres it hands Neo4j ids to the Postgres DAL and
  almost every read returns nothing. It must resolve by name against whichever
  backend is selected — which is only possible once terms, zones, and analyses
  exist in Postgres, i.e. after Phase 9.
* `gsf/dal/neo4j/datasources.py` formats the zone filter into a Cypher string
  and, under `GSF_STORE=postgres`, receives the Phase 5 predicate object and
  interpolates `None`. Harmless in production, where one backend is selected and
  the facade dispatches to it; fatal to any mixed-backend run.

Until both are fixed, **parity is asserted by hand-written per-phase tests, not
by the golden replay.** Say so plainly in the Phase 11 PR rather than implying a
green oracle that was never run against Postgres.

Then: default `GSF_STORE=postgres`, full manual pass, then one PR deleting:
`gsf/dal/neo4j/`, `neo4j_tx.py`, `cypher_fragments.py`, the `GSF_STORE` switch
(flatten `gsf/dal/pg/*` up a level), `neo4j>=6.1.0` from `pyproject.toml`,
`_check_neo4j` + `HealthResponse.neo4j` (`gsf/server/responses.py:159` — no
frontend consumer found), the `neo4j` service + 4 volumes in
`docker-compose.yml`, `helm/gsf/templates/neo4j.yaml`, the `wait-for-neo4j`
init container in `_helpers.tpl`, `NEO4J_*` from `values.yaml`/`secret.yaml`/
`.env.example`/`dev_tools/setup_env.sh`, and the README/DEPLOYMENT sections.
Rename lingering `neo4j_id` vocabulary in `gsf/semantic/semantic_fk.py`
(prompt text at `:50-55`, `:253-274`) and `gsf/semantic/models.py:191`.

---

## Keeping main green

**One `GSF_STORE` env var selecting a whole parallel implementation, flipped
once in Phase 11.** Per-domain flags are ruled out by evidence, not caution —
cross-domain reads are pervasive:

- `fetch_table_exploration_details` (`exploration.py:194`) — one Cypher `UNION`
  spanning `Table→Term` **and** `Table→Column→ColumnAttribute→Term`
- `fetch_data_exploration_graph` (`:626`) — `TABLE_COUNTS_SUBQUERY` counts
  `Column`, `Sql` **and** `Term` in one query
- `fetch_table_zones_map` (`:562`) — zones joined against `CONTAINS*0..2`
- same pattern at `terms.py:624,638,700,715` and `sql_attributes.py:276`

Any boundary you can draw is crossed by at least one existing query.

Wire the switch by module, not by `if` in ~250 function bodies:

1. `git mv gsf/dal/<domain>.py gsf/dal/neo4j/<domain>.py` — pure move, zero
   content change, trivially reviewable.
2. New impls land at `gsf/dal/pg/<domain>.py`.
3. `gsf/dal/<domain>.py` becomes a ~10-line selector with an **explicit**
   import list (not `import *` — the lists *are* the frozen public surface, they
   double as a porting checklist, and `import *` breaks the existing
   `@patch("gsf.dal.terms.get_neo4j_conn")` tests).

`GSF_STORE` is read once in `gsf/dal/_store.py`, and must be set identically in
all three processes — API, ingestion service, and the chat worker subprocess
(`gsf/server/chat/worker.py` spawns its own).

---

## Fixture databases

Today: `dev_tools/sql/testdb.sql`, 16 lines, 2 tables, 1 FK, 0 views. Every
claim this refactor makes about catalog fidelity would be validated against
that. Replace it.

**Primary — Pagila (Postgres).** The Postgres port of Sakila, pinned to
`pagila-v3.1.0` and trimmed. **Landed 2026-08-11** — actual measured shape, with
three of this section's original claims corrected (see below and
`dev_tools/sql/README.md`):

| Pagila feature | What it exercises |
|---|---|
| 7 views + 1 materialized view | `TableTypes` / `table_type` (`base table` / `view` / `materialized view`) — previously untested for anything but base tables |
| `payment` partitioned into 22 monthly children | partitions surface as tables in `information_schema`; whether the catalog should show them is a real question this forces us to answer |
| `film.special_features text[]`, `mpaa_rating` enum, `fulltext tsvector`, 2 domains incl. `bıgınt` | column `data_type` handling beyond scalars; non-ASCII identifiers |
| `film_actor`, `film_category` junction tables | bridge-table detection (`gsf/semantic/bridge_tables.py`), `fetch_bridge_table_candidates` |
| `customer→address→city→country` | multi-hop `find_join_path` traversal |
| 1000 films / 4581 inventory rows | `store_column_sample_values`, `store_column_uniqueness` — meaningless on 16 lines |
| two schemas (`public` + GSF-authored `analytics`), 2 cross-schema FKs | `Schema` tier, schema-scoped zones, and the `-Schema` exclusion in `find_join_path` |

> **Corrections (2026-08-11).** Three claims above were asserted before being
> checked and were wrong. Pagila has **no self-referencing FK** — there is no
> `staff.reports_to`; that's Sakila/Northwind. Upstream Pagila is
> **single-schema**, so the second schema is `dev_tools/sql/pagila_analytics.sql`,
> authored here. And it is **22 base tables, not 15** (the `payment` partitions).
> The self-reference case is covered by Chinook's `Employee.ReportsTo` and, in a
> non-public schema, by `analytics.category_rollup.parent_rollup_id`. All of
> these are now asserted by `dev_tools/tests/test_fixtures.py` rather than
> described in prose and trusted.

**Second — Chinook (SQLite).** Exists purely to make the fixture set
*multi-database and multi-dialect*: 11 tables, clean FKs, and it drives
`gsf/connectors/sqlite.py` and a second sqlglot dialect through the parsers.
Without a second database, these are untestable: cross-database rejection in
`find_join_path` (`attributes.py:404`), `delete_by_database` /
`delete_all_data(database_name)` scoping, per-database pgvector collection
resets, and zones spanning databases.

**Mechanics (as landed).** `dev_tools/sql/pagila.sql` (1.2 MB) and
`chinook.sql` (0.6 MB) are vendored; the Chinook `.sqlite` is *generated* from
the vendored SQL by `dev_tools/build_sqlite_fixtures.py` and gitignored, so git
stores diffable text rather than a 1 MB blob. `dev_tools/seed_fixtures.py` runs
both halves in one command. `dev_tools/sql/build_pagila.sh` regenerates the
Pagila dump so the pin is a one-command bump rather than a mystery blob.
Both are **MIT** (the plan said Pagila was BSD — it isn't), recorded in
`THIRD_PARTY_NOTICES.md`.

Four non-obvious things this ran into, all documented in
`dev_tools/sql/README.md`:

- **Pinned to `pagila-v3.1.0`, not `master`.** Master targets PG18 (`uuidv7()`
  defaults, `VIRTUAL` generated columns); GSF runs `pgvector/pgvector:pg17`.
  v3.1.0 is the newest tag that loads on PG17 *unmodified* and still has the
  partitioned table, both domains, the enum and all 8 views — pinning beats
  patching a third-party fixture.
- **Trimmed** `payment` then `rental` to 600 rentals, in that order so the FK
  never breaks: ~13 MB → 1.2 MB with every table, view, type and constraint
  intact, and zero orphans (asserted in the fixture tests).
- **Dumped as `INSERT`s** (`--rows-per-insert=200`), because the seed script
  executes through psycopg's `cur.execute()`, which cannot run `COPY ... FROM
  stdin`.
- **`\restrict`/`\unrestrict` stripped** — `pg_dump` 17.6+ emits psql
  meta-commands that are not SQL.

Keep `testdb.sql` as a fast smoke fixture for tests that don't need breadth.

---

## Testing

There are 23 test files today, **none** touching a database and **none**
touching a router. Greenfield means there is no production diff to fall back
on: tests are the only safety net, so they are a **deliverable of every phase,
not a phase of their own**. A ported DAL function without a test is not done.

### Standing rule per phase

For every function ported in Phases 5–10:

1. **Round-trip test** — write via the DAL, read via the DAL, assert the shape.
2. **Golden test** — output matches the Phase-2 capture from Neo4j, byte for
   byte after normalisation.
3. **Zone-scoped variant** — the same call under `zone_ids=None`, a scoped
   list, and `[]`. This is non-optional; it is the access-control boundary.
4. **Empty/missing case** — unknown id, empty table, null description. Neo4j
   returned `[]` where SQL will happily raise or return `None`.

### Infrastructure (Phase 3, before any read phase)

- **Real Postgres in CI via `testcontainers[postgres]`** — non-negotiable. The
  whole refactor is SQL correctness; mocking a `Connection` proves only that
  `.execute()` was called. Session-scoped container, `alembic upgrade head` on
  start, function-scoped transaction rollback per test so tests share one
  seeded fixture without cross-contamination. Fall back to a CI service
  container if the runner can't do docker-in-docker — the fixture layer
  shouldn't care which.
- **The repo's first `conftest.py`** — there is none anywhere today. Fixtures:
  `pg_engine` (session), `db` (function, rollback), `pagila_catalog` (session,
  ingested once), `chinook_catalog`, `semantic_layer`, `zones`, `api_client`.
  This is the single biggest investment and it amortises across all 16 modules.
- **Coverage gate** — `pytest-cov` with a floor on `gsf/dal/` and
  `gsf/catalog/`, raised phase by phase. Start at whatever Phase 5 lands and
  ratchet; a gate that only ever goes up is enough.

### Test suites to build

| Suite | Phase | What it covers |
|---|---|---|
| DAL surface parity | 0 | name-set + `inspect.signature` equality between `pg` and `neo4j` impls — catches the likeliest cutover failure, a renamed kwarg or dropped passthrough |
| Fork equivalence | 1 | forked writer vs. library writer produce an identical Neo4j graph (node counts per label, rel counts per type, sorted property dump) |
| Golden captures | 2 | every `fetch_*`/`get_*`/`list_*`, three zone modes each |
| Transaction semantics | 3 | commit, rollback, nesting reuses outermost, fallback outside a scope, contextvar cleared after failure, **concurrent access from threads and a spawned subprocess** (the pool must be safe for all three call contexts) |
| Alembic | 3 | `upgrade head` on blank DB; `alembic check` reports no drift; downgrade/upgrade round-trip; `prisma db push` still succeeds afterwards |
| Catalog ingest | 4 | full Pagila + Chinook ingest; re-ingest is idempotent; incremental diff (add/rename/drop a table and column) does the right thing; `table_type` correct for views and matviews |
| Per-module DAL | 5–10 | the standing rule above, ~16 modules |
| `find_join_path` | 7 | own suite: **directionality** (two columns `SEMANTIC_FK`→ the same attribute must return `[]`, not a fabricated join — the single most important case); cycle termination; `-Schema` exclusion (two columns in different schemas connected only via `Schema` → `[]`); shortest path wins and is deterministic on ties; hop pairing (a known 2-hop path → exactly 2 hop dicts, correct source/target); cross-database rejection via a shared `ColumnAttribute` across Pagila and Chinook — only testable because there are two fixture databases; depth bound; **BFS and recursive-CTE implementations agree on every fixture pair** |
| Router tests | 5+ | `TestClient(create_app())` against the seeded fixture — **every route, not just GETs**: 200 + response-model validation, 404 on unknown id, 403 on zone-scoped denial, and the mutation paths (`PATCH /nodes/{id}`, zone CRUD, term updates, connection create/delete). Zero coverage today. |
| Semantic pipeline | 7 | `/semantic/compile` end to end on Pagila with LLM calls stubbed — deterministic given fixed model output |
| Model interchange | 10 | export→import idempotency (second import: all skipped, zero created); `replace=True` scoped delete; import of a foreign `imported_id`; malformed YAML |
| Invariants | ongoing | exploration degree counts agree across endpoints; paging stable across `skip`/`limit`; disabled zone grants no access but stays admin-visible; `find_join_path` never crosses databases; every `Table` reachable from its `Database` |
| Reset | 8 | `delete_all_data` / `delete_semantic_layer` / `delete_data_layer` each leave *exactly* the intended rows, scoped and unscoped, across both fixture databases |

### Existing Neo4j-coupled tests

- `gsf/semantic/tests/test_neo4j_dal_merge.py` — **delete.** It asserts on
  Cypher substrings (`"MERGE (term:Term" in query`), so it tests the
  implementation being removed and cannot be ported. Replace its five
  behaviours with round-trip tests against the fixture — a net gain.
- `gsf/dal/tests/test_neo4j_tx.py` — rewrite against `gsf/dal/pg/session.py`,
  keeping all five cases, against real PG rather than `MagicMock`. Must land in
  the same commit as the seam swap.
- `test_model_interchange.py`, `test_bridge_tables.py` — keep, repoint the
  patches, convert to fixture-based in Phases 10 / 7.
- `test_worker.py`, `test_databricks_oauth.py` — incidental `"neo4j down"`
  strings; rename for hygiene in Phase 11.

---

## Verification

Per phase, as stated in the **Done** criteria above. End to end, after Phase 11:

```bash
# clean slate
docker compose down -v && docker compose up -d postgres
uv run alembic upgrade head
uv run --no-sync python -m dev_tools.seed_local_postgres   # pagila + chinook
cd frontend && pnpm prisma db push   # must still succeed against public

# backend + ingestion
uv run uvicorn gsf.server.__main__:app --reload --port 3001 --app-dir .
uv run python -m gsf.ingestion_service   # :3002

# full loop
curl -sf localhost:3001/api/health          # no neo4j field, 200
curl -X POST localhost:3001/api/connections -d @dev_tools/fixtures/pagila.json
curl -X POST localhost:3001/api/connections -d @dev_tools/fixtures/chinook.json
curl -X POST localhost:3002/semantic/compile
uv run pytest --cov=gsf/dal --cov=gsf/catalog   # incl. testcontainers suite
```

Then in the UI, against Pagila: browse the catalog tree down to columns,
confirm the views and materialized view show the right type, open the
exploration graph in both data and semantic modes, check a term's column
attributes, run a chat query end to end ("which films rented most in 2022"),
export the model to YAML and re-import it (must be idempotent), and delete the
Chinook connection — Pagila must be untouched.

Finally: `grep -rni neo4j gsf/ frontend/ helm/ docker-compose.yml` should
return nothing but historical comments.

```bash
cd frontend && pnpm lint && pnpm format
uv run ruff check gsf/ && uv run ruff format gsf/
```

---

## Effort

Estimates below **include** the tests each phase must ship, per the standing
rule — roughly a third of each read phase is test code.

| Phase | Days | Parallel with |
|---|---|---|
| 0 — land docs, sever constants, freeze surface | 2–3 | land first |
| 1 — fork write path | 6–9 | 2, 3 |
| 2 — fixture DBs + golden capture | 4–5 | 1, 3 |
| 3 — PG foundations + test harness | 6–8 | 1, 2 |
| 4 — catalog writes **(gate)** | 7–10 | — |
| 5 — users + zones | 3–4 | — (sets filter contract) |
| 6 — datasources | 6–8 | 7 |
| 7 — terms / attributes / sql_attributes | 11–16 | 6, 8 |
| 8 — analyses, candidates, connections, reset | 6–8 | 6, 7 |
| 9 — exploration | 6–8 | 10 |
| 10 — model_interchange | 8–12 | 9 |
| 11 — flip + delete | 2–3 | — |

**Serial ≈ 66–93 engineer-days. With 3 people ≈ 8–11 elapsed weeks**, dominated
by two unavoidable serial gates: Phase 4 (nothing readable until something
writes) and Phase 5 (the zone-filter contract every later read consumes).

The test investment is roughly +18–21 days over a no-tests port. That is the
price of the greenfield decision: with no production data to diff against and
zero DB or router coverage today, the alternative isn't a faster refactor — it's
an unverifiable one.
