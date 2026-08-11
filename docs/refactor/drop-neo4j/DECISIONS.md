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

_None yet._
