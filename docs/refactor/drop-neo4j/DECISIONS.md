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
