<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AGENTS.md — Auto Ontology agent entry point

Skills that teach an agent how to **set up** the current Auto Ontology implementation and
**use** its ontology layer. They live in **`skills/`** (repo root), one directory
per skill, under the durable `nvidia-ontology*` namespace.

This file is for developers bringing Auto Ontology up and partners embedding it in an
agent harness. Auto Ontology **maintainers** (build, test, schema, frontend conventions)
stay on [`CLAUDE.md`](./CLAUDE.md). Do not fold those conventions into these
skills.

## Which skill to read

- [`skills/nvidia-ontology-setup/`](./skills/nvidia-ontology-setup/) — local Compose, `--dev`,
  Helm pointer, MCP against an already-running instance, setup logging, and
  troubleshooting. Start here when Auto Ontology is not up yet or something failed
  during bring-up.
- [`skills/nvidia-ontology-management/`](./skills/nvidia-ontology-management/) — inspect, model,
  **modify**, and safely publish through glossary terms, SQL attributes,
  semantic relationships, model YAML import/export, and governed result
  definitions. MCP cannot write; this skill covers the approved write paths.
- [`skills/nvidia-ontology-query/`](./skills/nvidia-ontology-query/) — how an
  agent calls Auto Ontology (MCP first, REST fallback), discovers meaning through the
  semantic layer, and validates SQL, rows, and answers. Start here when Auto Ontology is
  already deployed.

## Workflow ownership and handoffs

| User action | Owning skill |
| --- | --- |
| Install, configure, connect, ingest, compile, prove readiness | `nvidia-ontology-setup` |
| Inspect meaning and semantic relationships | `nvidia-ontology-management` |
| Design concepts, mappings, relationships, measures, units, grain, and policies | `nvidia-ontology-management` → `references/modeling.md` |
| Apply, verify, certify honestly, publish, and roll back | `nvidia-ontology-management` → `references/publication.md` |
| Ask descriptive or diagnostic questions and validate SQL, rows, and prose | `nvidia-ontology-query` → `references/query-validation.md` |

The composed lifecycle is install/connect/readiness → inspect/model → staged
apply/readback → query validation → governed publication → fresh query readback.
A step that lacks product support must return an explicit gap or receipt rather
than inventing an operation. Authentication, permissions, provenance, and error
handling remain shared concerns rather than separate overlapping skills.

## Sources of truth (do not duplicate)

Skills encode **workflow, order, and pitfalls**. They must not become a second
API catalog or a second installer.

| Job | Owner | Skills do |
| --- | --- | --- |
| Install commands | `dev_tools/setup_env.sh`, `docker-compose.yml`, `DEPLOYMENT.md` | Invoke them; log what ran |
| Live read tools | `mcp/auto_ontology_mcp/tools.py` (handshake tool list) | Prefer MCP; sequence reads; never copy the tool table |
| REST shapes and permissions | `docs/openapi/auto-ontology-api.json` | Name the operation and permission; link; do not paste schemas |

MCP is read-only by design. Writes go through the Next.js `/api/...` surface.
The live MCP tool list at handshake is the source of truth if it disagrees
with anything in a skill.

## Catalog mirror (not done in this repo)

These directories are shaped for a later sync to
[github.com/nvidia/skills](https://github.com/nvidia/skills) (`npx skills add nvidia/skills`)
and build.nvidia.com/skills. That pipeline requires, per skill:

- `SKILL.md`, `skill-card.md`, `evals/evals.json` (present here)
- `skill.oms.sig` — NVIDIA OMS signature (not generated here; do not fabricate)
- a Tier-3 eval run that fills the skill-card tables (representative pilots are recorded; repeated full-matrix runs remain pending)
- a `components.d/nvidia-ontology.yml` entry in **nvidia/skills** registering:

```yaml
name: Auto Ontology
repo: NVIDIA/auto-ontology
skills:
  - path: skills/nvidia-ontology-setup/
    catalog_dir: nvidia-ontology-setup
  - path: skills/nvidia-ontology-management/
    catalog_dir: nvidia-ontology-management
  - path: skills/nvidia-ontology-query/
    catalog_dir: nvidia-ontology-query
```

Until that lands, agents use the copies in this checkout. Skills must stay
free of staging hostnames, cluster names, Vault paths, and credentials —
the catalog is public even when Auto Ontology itself is access-restricted.
