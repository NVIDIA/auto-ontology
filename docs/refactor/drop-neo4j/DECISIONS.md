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
