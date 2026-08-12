# Drop Neo4j — decision records

Numbered records for anything **not** already settled in [PLAN.md](PLAN.md),
and for every deviation from it.

A record must land **before** the code it justifies. Then amend `PLAN.md` in the
same commit, so the two never disagree.

**Format:**

```
## NNN — <short title>

**Date:** YYYY-MM-DD  **Phase:** N  **Supersedes:** PLAN.md §<section> | DECISION-NNN | nothing

**Context** — what forced the decision.
**Decision** — what was chosen.
**Consequences** — what this costs, what it rules out, what has to change elsewhere.
```

---

## Known behaviour changes

These are accepted, deliberate divergences from current Neo4j behaviour. Each
must be restated in the PR description of the change that introduces it.

| # | Change | Introduced in |
|---|---|---|
| B1 | Reset deletes become **narrower** than `apoc.path.subgraphNodes`, which today bleeds across databases through shared `Term`/`Sql` nodes | Phase 8 |
| B2 | `_delete_semantic_nodes`' scoped branch does **not** collect `PqlAnalysis` — current behaviour, deliberately preserved rather than "fixed" | Phase 8 |
| B3 | Postgres catalogs gain partitioned parent tables and materialized views, which were silently dropped before; 5 columns per Pagila report their declared type (`mpaa_rating`, `text[]`, `year`) instead of `USER-DEFINED`/`ARRAY` | Phase 2 ([003](#003--fix-partitioned-tables-and-materialized-views-missing-from-postgres-catalogs)) |
| B4 | The source connection `run_ingest` holds open now covers extraction only, not extraction + embedding — strictly narrower, and measurably so on Databricks | Phase 1 ([004](#004--the-shape-of-the-fork-boundary)) |
| B5 | `SemanticEmbedder` builds its one-node embed graph per call instead of once in `__post_init__`; the `embed_graph` attribute is gone | Phase 1 ([004](#004--the-shape-of-the-fork-boundary)) |
| B6 | **Column-level changes now apply on re-ingest.** Added columns appear, dropped columns are removed, changed types update. None of this happened before — the column diff raised on every run and the error was swallowed | Phase 4 ([006](#006--column-diffs-never-ran-and-the-error-was-swallowed)) |
| ~~B7~~ | **Withdrawn.** Ranking the suggester *would* have been a behaviour change, so it is not being made here — the bug is documented in place and the inert behaviour preserved. See [007](#007--the-sqlattribute-suggester-was-never-actually-ranking) |

Any further behaviour change discovered mid-port gets added to this table
**and** its own numbered record below. Absorbing one silently is the single
highest risk of unattended execution.

---

## Records

## 001 — Merge `origin/main` at every phase boundary

**Date:** 2026-08-11  **Phase:** 0  **Supersedes:** nothing (adds to PLAN.md § This plan lives in the repo)

**Context** — The plan set out git conventions but said nothing about staying
current with `main`. The refactor runs 8–11 weeks against an actively developed
branch and concentrates on `gsf/dal/`, which most feature work also touches. A
long-lived branch that only merges at the end would face conflict resolution
between ported SQL and upstream Cypher edits in files whose two versions no
longer resemble each other — and the failure mode is silent: resolving with
"ours" drops upstream behaviour that was never ported.

**Decision** — Every phase starts with `git fetch origin && git merge
origin/main`, before any code is written. The merge is logged in `PROGRESS.md`
as the phase's first entry, recording the merged SHA and whether it conflicted.
Upstream Cypher arriving in an already-ported module requires its own decision
record describing how the behaviour was carried into the Postgres
implementation.

**Consequences** — Conflicts stay small and land at the one moment when nothing
is half-ported. Costs a few minutes per phase and occasionally forces a port to
be redone against changed upstream behaviour, which is the point: that work is
real either way, and it is far cheaper discovered at a phase boundary than at
the Phase 11 flip.

**Two-way ambiguity worth naming:** during Phases 5–10 an upstream change to a
DAL function's *signature* will fail `test_dal_surface.py` on the merge, not on
the port. That is intended — it is the freeze doing its job — but the fix is to
regenerate the snapshot and carry the change into both implementations, never
to revert the upstream change.

---

## 002 — Pagila pinned to v3.1.0, trimmed, and extended with a GSF-authored schema

**Date:** 2026-08-11  **Phase:** 2  **Supersedes:** PLAN.md § Fixture databases (amended in the same commit)

**Context** — The plan specified "vendor Pagila (≈3 MB, BSD) and Chinook (MIT),
extend `seed_local_postgres.py`". Four things turned out differently once the
fixture was actually built:

1. Upstream `master` targets PostgreSQL 18 — `uuidv7()` column defaults and
   `VIRTUAL` generated columns. GSF runs `pgvector/pgvector:pg17`; the schema
   will not load.
2. Master's data file is 13 MB, not ~3 MB.
3. `pg_dump`'s default `COPY ... FROM stdin` cannot be executed through
   psycopg's `cur.execute()`, which is what the seed script uses; and
   `pg_dump` 17.6+ emits `\restrict`/`\unrestrict` psql meta-commands that are
   not SQL.
4. Upstream Pagila is single-schema and has no self-referencing foreign key, so
   two things the plan wanted it for — the Schema tier and `find_join_path`'s
   `-Schema` exclusion — were not actually testable with it.

**Decision** —
- Pin to tag `pagila-v3.1.0`, the newest that loads on PG17 **unmodified**,
  rather than patching master's PG18 features. It retains the partitioned
  table, both domains, the enum and all 8 views.
- Trim `payment` then `rental` (in that order) to 600 rentals: 13 MB → 1.2 MB
  with every table, view, type and constraint intact.
- Dump with `--inserts --rows-per-insert=200` and strip the psql
  meta-commands.
- Add `dev_tools/sql/pagila_analytics.sql`, a GSF-authored second schema with
  two tables, a view, two cross-schema FKs and a self-referencing FK.
- Split the seeder: `seed_local_postgres.py` keeps the Postgres fixtures,
  `build_sqlite_fixtures.py` builds Chinook, `seed_fixtures.py` runs both. The
  plan said to extend the existing loop; SQLite shares nothing with Postgres
  seeding beyond the word "fixture", so forcing it in would have been worse.
- Vendor `build_pagila.sh` so the dump is reproducible and the pin is a
  one-command bump.
- Pagila is **MIT**, not BSD as the plan stated.

**Consequences** — The fixture is a derived artifact, not a verbatim upstream
file, so its provenance has to be documented or it becomes a mystery blob;
`dev_tools/sql/README.md` carries that, and `build_pagila.sh` makes it
reproducible. Pinning to v3.1.0 means the fixture will drift from upstream
Pagila over time; that is preferable to carrying local patches, and the pin
moves when GSF's Postgres major does. The generated `.sqlite` is gitignored, so
a fresh checkout must run the seeder before SQLite-backed tests will pass —
`dev_tools/tests/test_fixtures.py` builds it on demand, so this is invisible in
practice.

**Behaviour-change watch — resolved same day by [003](#003--fix-partitioned-tables-and-materialized-views-missing-from-postgres-catalogs).**
This record originally flagged partitioned `payment` as a question for Phase 4:
do partition children belong in the catalog? Ingesting the fixture answered it
immediately and differently than expected — the children were already excluded
*and so was the parent*, so the table was absent altogether. That is a bug, not
a policy, and 003 fixes it: parent in, children out.

---

## 003 — Fix partitioned tables and materialized views missing from Postgres catalogs

**Date:** 2026-08-11  **Phase:** 2  **Supersedes:** nothing

**Context** — The Pagila fixture landed in Phase 2 specifically to surface
catalog shapes nothing previously tested. It did so immediately: ingesting
Pagila wrote 21 of the 30 relations in `public` to the graph. Missing were the
partitioned table `payment`, its 7 partition children, and the materialized
view `rental_by_category`. Two independent causes in
`gsf/connectors/postgres.py`:

1. `get_tables` allowed `relkind IN ('r','v','m','f')`. `relispartition = false`
   correctly hides partition *children*, but the parent is `relkind = 'p'`,
   which was not allowed through — so **neither** parent nor children appeared
   and an entire queryable table was invisible. The comment said the intent was
   to "filter tables that are part of partitioned tables", so this reads as an
   oversight rather than a decision.
2. Both `get_tables` and `get_columns` drove from `information_schema`, which
   does not list materialized views or their columns at all. The
   `relkind = 'm'` branch and `TableTypes.MATERIALIZED_VIEW` were therefore
   unreachable — dead code since they were written.

Both are pre-existing product bugs, not refactor artifacts.

**Decision** — Fix now, in Phase 2, rather than deferring. Drive both methods
from `pg_catalog` instead of `information_schema`, add `'p'` to the allowed
`relkind` set, and keep `relispartition = false` so children stay hidden.

Fixing now is safe *because the connector is not part of the port*.
`gsf/connectors/` sits above the DAL and is untouched by the Neo4j-to-Postgres
swap, so this changes both paths identically and cannot complicate Phases 5–10.
Deferring would have been worse: the golden captures are taken from the current
Neo4j behaviour and become the fidelity oracle for the entire refactor. Capture
them first and the bug is frozen into the oracle, and every later phase would
faithfully reproduce a catalog with missing tables.

**Consequences** — Behaviour change **B3**. On any Postgres source:

- Partitioned parent tables now appear in the catalog. They will be ingested,
  described and embedded, which is the point — they were unqueryable before.
- Materialized views now appear, typed `materialized view`. This is the first
  time that `TableTypes` value is ever produced, so any consumer that assumed
  only `base table`/`view` now sees a third value. None was found, but this is
  the risk worth naming.
- `data_type` changes for enum, array and domain columns: `USER-DEFINED` →
  `mpaa_rating`, `ARRAY` → `text[]`, `integer` → `year`. Strictly more useful
  to a SQL-generating model, but descriptions and embeddings for those columns
  will differ from what is currently stored, so they change on next ingest.
  Measured on Pagila: 5 of 143 previously-ingested columns.
- `format_type(atttypid, NULL)` was chosen precisely because it reproduces
  `information_schema.columns.data_type`'s unqualified spelling
  (`character varying`, not `character varying(255)`), so the other 138 columns
  are byte-identical. Verified, not assumed.

Covered by `gsf/connectors/tests/test_postgres.py` (8 tests), which needs the
Pagila fixture and skips without it.

---

## 004 — The shape of the fork boundary

**Date:** 2026-08-11  **Phase:** 1  **Supersedes:** PLAN.md § Phase 1 (amended in the same commit)

**Context** — PLAN.md § Phase 1 says "verbatim fork first, rewrite in Phase 4",
and lists the file layout. Four things it did not settle came up while doing it,
each of which is a place where "verbatim" and "GSF-owned" pull in opposite
directions.

1. **`extract_tabular_db_data(params)` took a `TabularExtractParams`** — a
   library type — and read exactly one attribute off it, `.connector`. Copying
   that signature verbatim would put a library import in `gsf/catalog/`, in a
   module that is otherwise storage- **and** library-agnostic and is meant to
   survive Phase 4 untouched.
2. **`store_relational_db_in_neo4j(data, dialect, num_workers)`** was a
   two-line forwarder to `populate_tabular_data`, plus a Neo4j-specific name on
   a function that Phase 4 makes write Postgres.
3. **The `_shared_connection` window.** `run_ingest` held the source database
   connection open for the whole `Graph()` chain, because all three operators
   ran inside `graph.execute(None)`. Straight-line calls make the window
   explicit, and there is nothing to hold it open *for* past extraction.
4. **`SemanticEmbedder.embed_graph`.** PLAN.md asks for the `_BatchEmbedActor`
   dependency to live in "one line in one file". `gsf/semantic/embed.py` was a
   second importer, and it cached the one-node graph on the dataclass.

**Decision** —

1. `extract_tabular_db_data` takes the **connector**. `TabularExtractParams`
   stays on the library and out of `gsf/catalog/` entirely.
2. `store_relational_db_in_neo4j` is **deleted**;
   `gsf.catalog.ingest.ingest_catalog` calls
   `gsf.catalog.write.populate_tabular_data` directly. This is the same call
   `TabularSchemaExtractOp` made, one indirection shorter.
3. `_shared_connection` wraps **`ingest_catalog` only**. Behaviour change
   **B4**.
4. `gsf/semantic/embed.py` goes through `gsf.utils.embedding.batch_embed` and
   the `embed_graph` field is dropped, so the graph is constructed per call.
   Behaviour change **B5**.

`gsf/catalog/store/connection.py` deliberately keeps the name
`get_neo4j_conn` and the module-level `_conn` singleton. Renaming it would have
touched 11 modules for a name Phase 11 deletes, and `gsf.dal.close_store()`
already exists as the seam Phase 3 repoints.

**Consequences** — 1 and 2 make `gsf/catalog/` importable without
`nemo_retriever` on the path for everything except the `SQLDatabase` type hint,
which is under `TYPE_CHECKING`. They also mean `gsf/catalog/extract.py` is *not*
byte-comparable to upstream, so `gsf/catalog/tests/test_fork_parity.py` lists it
under `DIVERGED` with this record's number rather than pinning it — every other
forked module is pinned, AST-for-AST.

3 is strictly narrower and strictly better: the connection is released before
the embed HTTP round trip rather than after, which on Databricks (where opening
one is the slowest and flakiest step, per `_shared_connection`'s docstring) is
the difference between holding a fragile connection for seconds and holding it
for minutes. It is still a change in when the source database sees a
disconnect, so it is named rather than absorbed.

4 constructs a `Graph` and an operator per `embed_term` call — measured at
microseconds, against an HTTP embed call in the same function, so the cost is
not observable. `SemanticEmbedder.embed_graph` is gone; `embed_all_semantic_nodes`
was the one external reader and now calls `batch_embed` itself.

**Not changed, deliberately:** `TabularFetchEmbeddingsOp`, `IngestVdbOperator`,
`SQLDatabase`, `EmbedParams` and `TabularExtractParams` all stay on the library,
exactly as PLAN.md § Scope requires. `TabularFetchEmbeddingsOp` is now *called*
directly instead of run through a `Graph()`, which is equivalent — `Graph`
invokes `operator.run(data)` and `AbstractOperator.__call__` is `run`. That
equivalence does **not** hold for `_BatchEmbedActor`, which is an archetype
operator resolved to a CPU or GPU variant only during graph execution; hence
`batch_embed` keeps the one-node graph.

## 005 — Split the column↔attribute link table, and rename the traversal view

**Date:** 2026-08-11  **Phase:** 3  **Supersedes:** the ERD as first landed in Phase 3 (revision `75bdf1cdf36d`, replaced by `96b629fa2ae5`)

**Context** — Raised at the ERD review gate. Two objections to the schema as
first written, both correct:

1. `HAS_ATTRIBUTE` and `SEMANTIC_FK` were one table, `column_attribute_link`,
   discriminated by `kind`.
2. The `join_edge` view documentation described `SEMANTIC_FK` as "outgoing
   only" as though that were a property of the edge.

The second was the more serious error. Checking the code:
`gsf/dal/attributes.py:151` (`fetch_attr_column_contexts`) binds a
`ColumnAttribute` and then matches
`(col:Column)-[:SEMANTIC_FK|HAS_ATTRIBUTE]->(attr)` — a **reverse** lookup,
finding the columns that reference a given attribute. So `SEMANTIC_FK` is
*stored* in one direction but *read* in both. The outgoing-only restriction
belongs to `find_join_path`'s traversal alone, and stating it as a property of
the edge would have led whoever reused that view for another traversal to
silently lose half the edges.

**Decision** —
- Split into `column_has_attribute` and `column_semantic_fk`. They assert
  different things (*is an instance of* vs *references*), their cardinality
  already differs — `resolve_semantic_fks` treats a column as unlinked when it
  has no `SEMANTIC_FK`, so at most one is expected, whereas a column can carry
  several attributes — and a discriminator inside the primary key makes any
  constraint that applies to only one of them impossible to express.
- Rename `join_edge` → `join_path_edge`, named for the single function it
  serves, and document that its one-way `SEMANTIC_FK` is a path-finding rule
  rather than a fact about the edge, with an explicit pointer to
  `column_semantic_fk` for reverse traversal.
- Index `attribute_id` on both link tables. The composite primary key already
  covers the `column_id` direction; the reverse direction is what
  `fetch_attr_column_contexts` needs.

**Consequences** — 24 tables instead of 23. The view gains one `UNION ALL`
branch and loses two `WHERE kind = ...` filters, so it is marginally simpler.
Phases 5–10 write against two narrow tables instead of one wide one, which is
what makes a future constraint on either side possible. Migration `75bdf1cdf36d`
was deleted and regenerated as `96b629fa2ae5` rather than superseded by a second
revision — it had never been applied outside a throwaway database, so there was
nothing to migrate *from*.

**Worth carrying forward:** this is the ERD gate working exactly as PLAN.md
intended. Both corrections were cheap here and would have been expensive after
Phase 7 wrote queries against the wrong shape.

---

## 006 — Column diffs never ran, and the error was swallowed

**Date:** 2026-08-11  **Phase:** 4  **Supersedes:** nothing

**Context** — Phase 1 flagged that `write.py`'s incremental diffing had no
coverage beyond the end-to-end fixture ingest, and asked for tests *before* the
Postgres rewrite. Writing them surfaced two stacked bugs that together mean
**no column-level change has ever been applied on a re-ingest**.

1. `update_diff_from_existing_schema` merges the existing and new `columns_df`
   on `["database", "schema", "table_name", "column_name"]`. Measured, the two
   frames share twelve columns — but **neither has `schema`** (both call it
   `table_schema`), and **only the graph side has `database`**. Every one of the
   three merges therefore raised `KeyError: 'database'`, and the function's
   `except` wrapped and re-raised it.
2. `populate_db` runs those updates through `executor.map(...)` and **never
   consumes the returned iterator**. `ThreadPoolExecutor.map` stores a worker's
   exception in its future and re-raises only on consumption, so the error was
   discarded in silence. `with ThreadPoolExecutor(...)` waits for completion but
   does not re-raise.

The visible symptom is that table add/delete worked (it happens *before* the
raise) while nothing column-level did. A column added to an existing table
never reached the catalog; a dropped column left a ghost the SQL generator
would still write queries against; a changed type stayed stale — with no log
line, no error, and a scheduler cheerfully reporting success every 24 hours.

Both bugs are upstream NeMo-Retriever's; the fork is verbatim.

**Decision** — Fix both, in the Neo4j implementation, now.

Reproducing this in the Postgres implementation for the sake of "behaviour
preservation" would be the wrong reading of that principle. Preserving
behaviour means preserving what the system is *meant* to do where the two
agree; it does not mean porting silent data loss so the two stores can be
wrong identically. Fixing it in the Neo4j path first also means the goldens and
the incremental tests describe one behaviour rather than two.

- `_COLUMN_MERGE_KEYS = ["table_schema", "table_name", "column_name"]` — the
  three keys both frames actually share. `database` is redundant: a `Schema`
  belongs to exactly one database.
- `list(executor.map(...))` in `populate_db`, so a failing schema update raises
  rather than vanishing.

**Consequences** — Behaviour change **B6**. On the next re-ingest of any
existing database, column additions, deletions and type changes that have
accumulated since the catalog was first built will all apply at once. For a
database whose schema has moved since ingestion, that is a large one-time diff —
correct, but worth expecting rather than being surprised by.

Consuming the executor's results also means a schema that fails to update now
**fails the ingest** instead of being skipped. That is the point, but it does
convert a silent partial success into a loud failure, and a source database with
one pathological schema will now surface it.

Both files leave the verbatim-fork set and are declared in `DIVERGED` with a
pointer here. Covered by `gsf/catalog/tests/test_incremental_ingest.py` — 12
tests over a throwaway source database: first ingest, unchanged re-ingest is a
true no-op (compared on node counts, since a duplicate-creating bug preserves
every name), table add/drop, column add/drop, rename as drop+add, type change
in place with the column keeping its id, add-then-drop leaving no trace, new
schema picked up, and foreign keys surviving a re-ingest.

---

## 007 — The SqlAttribute suggester was never actually ranking

**Date:** 2026-08-11  **Phase:** 4  **Supersedes:** nothing

**Context** — Porting `store/pg/queries.py` required deciding how to store the
per-month query counters (`count_8_2026`, one property per month a query was
seen), since dynamically-named properties have no relational equivalent. The
options were a child table or a `jsonb` blob, and the question was raised
whether they are used at all.

Tracing the consumer chain:

```
Query.__init__                    writes count_{month}_{year}
update_counters_...               increments it
fetch_terms_with_sqls             returns all Sql properties
  └─ sql_attribute_suggester:562
       └─ _rank_expressions       :427
            └─ _latest_3month_score
```

So they are read — the suggester ranks candidate SQL expressions by them, which
decides what the semantic layer proposes as SqlAttributes.

Except the read never worked. `_latest_3month_score` matched keys against
`^count_monthly_(\d{4})_(\d{2})$`, while `Query.__init__` writes
`count_{month}_{year}` — `count_8_2026`. **Nothing has ever written a name that
regex accepts.** Verified directly: the scorer returns `0.0` for a real property
bag from the fixture, and `42.0` only for a hand-made key nothing produces.

Every expression therefore scored 0.0, `sorted(..., reverse=True)` on equal
values is a no-op, and the ranking has always been dict insertion order.

Two further faults were latent behind it, and are worth recording because they
would have been inherited by any faithful port:

- The sort key was `(int(m.group(1)), int(m.group(2)))` — the regex's groups are
  `(year, month)` but the writer's format is `month_year`, so had the names ever
  matched, the ordering would still have been wrong.
- `fetch_terms_with_sqls`'s docstring documents the property as
  `count_monthly_YYYY_MM`, matching the broken regex rather than the writer.
  The docstring is where the mistake most likely started.

**Decision** — Two parts, and the second was a correction.

**Drop per-month counters from the schema.** Nothing can read them, and the
`total_counter` that would replace them is already a single integer column. This
settles the question that prompted the investigation: `sql_query` needs five
scalar columns, no child table, no `jsonb`, no dynamic property names.

**Do not turn ranking on. Preserve the inert behaviour, documented in place.**
The fix was written and then deliberately withdrawn. Ranking the suggester
changes which SqlAttributes the semantic layer proposes — a product behaviour
change, in a component whose output this refactor is meant to leave untouched.
It belongs in its own change with its own tests, not smuggled in as a side
effect of a port.

`_usage_score` therefore returns a literal `0.0`, with the cause, the two latent
faults behind it, and the one-line fix in its docstring. A constant is honest
where a plausible-looking scorer that silently does nothing is not.

**This is the opposite call to [006](#006--column-diffs-never-ran-and-the-error-was-swallowed),
and the distinction is the point.** That bug silently *lost data* — columns
vanished from the catalog — so reproducing it faithfully would have meant
porting data loss. This one produces an arbitrary ordering of suggestions.
Wrong, but not destructive, and safe to leave until it can be changed on
purpose.

**Consequences** — No behaviour change; **B7 is withdrawn**. The bug stays open,
recorded in PLAN.md's bug table alongside the `is_nullable` export fault.

`gsf/semantic/tests/test_usage_score.py` pins the inertness, so a change to it
is visible rather than silent — and says explicitly that every assertion there
inverts when the fix lands. Nothing caught this for the life of the feature
because no test asserted anything at all about the score.

---

## 008 — The zone-filter contract: a predicate, not a WHERE string

**Date:** 2026-08-11  **Phase:** 5  **Supersedes:** nothing (settles a question PLAN.md § Phase 5 raises)

**Context** — `resolve_table_filter` is how nearly every read in the DAL narrows
itself to what a user may see. The Neo4j version returns
`(where_clause_string, params)` for callers to interpolate into an f-string
query. PLAN.md flags deciding its Postgres shape as the thing every later read
phase depends on: "decide once here or every later phase re-litigates it."

The obstacle is the surface freeze, which requires both implementations to
present identical signatures — and a SQLAlchemy predicate is not a string.

What settles it is who calls it. Measured: `resolve_table_filter` is used in
`terms`, `exploration`, `sql_attributes`, `datasources`, `custom_analyses` and
`cypher_fragments` — **all inside `gsf/dal`**, each ported with its own phase.
No caller outside the DAL exists, so the two shapes never meet. By contrast
`resolve_accessible_catalog_ids` *is* called from `gsf/server/terms/service.py`,
and returns plain id sets, which are backend-neutral.

**Decision** — The Postgres `resolve_table_filter` takes a SQLAlchemy column and
returns `(predicate | None, params)`. `params` is kept and passed through
unchanged for signature compatibility; Core binds its own parameters, so nothing
is added to it.

Returning a SQL string instead would have preserved the signature exactly, but
it would also have discarded the reason Core was chosen over raw SQL — safe
composition of conditional filters, which is precisely what this function is for.

`test_dal_surface.py` gains a `BACKEND_SPECIFIC` allowlist: annotations may
differ where declared, **parameter names may not**, because those are what
callers pass and a renamed keyword breaks a caller whatever the types say.
Adding to that list requires a record here first.

**Consequences** — Phases 6–10 write `.where(predicate)` rather than
interpolating a clause, which is the shape the rest of the port wants anyway.
The allowlist is a hole in the freeze and is deliberately narrow: one entry, one
reason, and parameter names still enforced.

**Named explicitly, because it is an access-control trap:** `zone_ids=None`
means *unscoped* — an admin seeing the whole catalog — while `[]` means *nothing
is granted*. Conflating them turns a locked-down user into an administrator.
Both implementations are covered by tests that assert the two produce different
answers.

---

## 009 — The five semantic-dependent `datasources` functions move to Phase 7

**Date:** 2026-08-12  **Phase:** 6  **Supersedes:** PLAN.md § Phase 6 and § Phase 7 (both amended in the same commit)

**Context** — `datasources` has 26 public functions. **19 read only the catalog
tier**, which Phase 4 already populates in Postgres. **Seven** reach the
semantic tier: `fetch_tables_for_schema`, `fetch_all_tables_without_term`,
`fetch_columns_for_table`, `fetch_tables_and_columns_by_node_ids`,
`fetch_bridge_table_candidates`, `fetch_tables_by_ids` and
`fetch_table_context`.

**Five of the seven reach it the same way** — through
`column_description_expr` / `table_description_expr`, the fallback that reads a
description off a linked `ColumnAttribute` or `Term` when the catalog node has
none. Only `fetch_all_tables_without_term` and `fetch_bridge_table_candidates`
depend on the semantic tier for anything else. So this is largely *one* concept
threaded through many reads, and porting the description fallback into
`sql_fragments` unblocks most of the group at once.

> **Corrected twice before implementation: 21/5, then 20/6, then 19/7.**
>
> The first classifier resolved query constants with a regex that stopped at the
> first line beginning with a letter — and every one of these constants is a
> Cypher string whose second line is `MATCH`, so it read one line of each and
> found nothing. The second recursed but guarded self-reference with "the text
> does not start with its own name", true of *every* assignment, so the
> recursion never ran and it reproduced the first answer exactly. The third pass
> was hand-checking: it caught `fetch_tables_by_ids` and then
> `fetch_table_context`, but had no principled reason to stop there.
>
> Two independent bugs agreeing on the same number is how a wrong figure
> survives review; hand-checking after that is how it survives a second time.
> The classifier is now committed as `dev_tools/classify_dal_dependencies.py`
> and prints the chain proving each verdict, so the next phase checks the answer
> instead of trusting it.

Those five are implementable now, since the tables exist from Phase 3. They are
not **verifiable** now. Nothing writes terms or attributes until Phase 7, so
every semantic join returns empty — and against no data a wrong join is
indistinguishable from a correct one. That is the exact failure mode this
refactor has already been caught by three times: a reader and a writer
disagreeing, with no data present to make the disagreement visible.

Their golden captures are non-trivial precisely because the fixture *does* have
semantic data, so the check that would catch a mistake is the one that cannot
run yet.

**Decision** — Phase 6 implements and verifies the 19 catalog-tier functions.
The seven that reach the semantic tier move into Phase 7, where their data and
their goldens arrive together.

The alternative was to implement all 26 now and re-verify five of them during
Phase 7. Rejected: it leaves five functions carrying unearned confidence in the
interval, and "verified later" is a promise the next person has to remember. A
phase that ends with everything it produced actually checked is worth more than
one that ends on the original schedule.

**Consequences** — `datasources` is finished across two phases, so
`gsf/dal/pg/datasources.py` is incomplete at the end of Phase 6 and the
selector will raise for those seven under ``GSF_STORE=postgres``. That is
deliberate and loud: a ``NotImplementedError`` naming the phase is better than a
query that quietly returns nothing.

Phase 7 grows by seven functions and keeps its estimate — they are small, and
their cost was always going to be the semantic fixture rather than the SQL.

**Phase 6's Done criteria change accordingly:** `validate_sql` resolving against
a Postgres-built `Schema` map still belongs to Phase 6, because
`fetch_schemas_by_ids` is catalog-only and is the function that feeds it.
