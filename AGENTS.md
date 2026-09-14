<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AGENTS.md — NVIDIA Ontology agent entry point

Skills that teach an agent how to **set up** the current GSF implementation and
**use** its ontology layer. They live in **`skills/`** (repo root), one directory
per skill, under the durable `nvidia-ontology*` namespace.

This file is for developers bringing GSF up and partners embedding it in an
agent harness. GSF **maintainers** (build, test, schema, frontend conventions)
stay on [`CLAUDE.md`](./CLAUDE.md). Do not fold those conventions into these
skills.

## Which skill to read

- [`skills/nvidia-ontology-install/`](./skills/nvidia-ontology-install/) — local Compose, `--dev`,
  Helm pointer, MCP against an already-running instance, install logging,
  troubleshooting. Start here when GSF is not up yet or something failed
  during bring-up.
- [`skills/nvidia-ontology/`](./skills/nvidia-ontology/) — inspect, model,
  **modify**, and safely publish through glossary terms, SQL attributes,
  semantic relationships, model YAML import/export, and governed result
  definitions. MCP cannot write; this skill covers the approved write paths.
- [`skills/nvidia-ontology-agent/`](./skills/nvidia-ontology-agent/) — how an
  agent calls GSF (MCP first, REST fallback), discovers meaning through the
  semantic layer, and validates SQL, rows, and answers. Start here when GSF is
  already deployed.

## Workflow ownership and handoffs

| User action | Owning skill |
| --- | --- |
| Install, configure, connect, ingest, compile, prove readiness | `nvidia-ontology-install` |
| Inspect meaning and semantic relationships | `nvidia-ontology` |
| Design concepts, mappings, relationships, measures, units, grain, and policies | `nvidia-ontology` → `references/modeling.md` |
| Apply, verify, certify honestly, publish, and roll back | `nvidia-ontology` → `references/publication.md` |
| Ask descriptive or diagnostic questions and validate SQL, rows, and prose | `nvidia-ontology-agent` → `references/query-validation.md` |

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
| Live read tools | `mcp/gsf_mcp/tools.py` (handshake tool list) | Prefer MCP; sequence reads; never copy the tool table |
| REST shapes and permissions | `docs/openapi/gsf-api.json` | Name the operation and permission; link; do not paste schemas |

MCP is read-only by design. Writes go through the Next.js `/api/...` surface.
The live MCP tool list at handshake is the source of truth if it disagrees
with anything in a skill.

## Catalog mirror (not done in this repo)

These directories are shaped for a later sync to
[github.com/nvidia/skills](https://github.com/nvidia/skills) (`npx skills add nvidia/skills`)
and build.nvidia.com/skills. That pipeline requires, per skill:

- `SKILL.md`, `skill-card.md`, `evals/evals.json` (present here)
- `skill.oms.sig` — NVIDIA OMS signature (not generated here; do not fabricate)
- a Tier-3 eval run that fills the skill-card tables (not run here)
- a `components.d/nvidia-ontology.yml` entry in **nvidia/skills** registering:

```yaml
name: NVIDIA Ontology
repo: NVIDIA/GSF
skills:
  - path: skills/nvidia-ontology-install/
    catalog_dir: nvidia-ontology-install
  - path: skills/nvidia-ontology/
    catalog_dir: nvidia-ontology
  - path: skills/nvidia-ontology-agent/
    catalog_dir: nvidia-ontology-agent
```

Until that lands, agents use the copies in this checkout. Skills must stay
free of staging hostnames, cluster names, Vault paths, and credentials —
the catalog is public even when GSF itself is access-restricted.
