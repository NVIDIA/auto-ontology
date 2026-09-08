<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AGENTS.md — GSF AI agent entry point

Skills that teach an agent how to **set up** GSF and **use** its ontology
layer. They live in **`skills/`** (repo root), one directory per skill, named
`gsf-*`.

This file is for developers bringing GSF up and partners embedding it in an
agent harness. GSF **maintainers** (build, test, schema, frontend conventions)
stay on [`CLAUDE.md`](./CLAUDE.md). Do not fold those conventions into these
skills.

## Which skill to read

- [`skills/gsf-install/`](./skills/gsf-install/) — local Compose, `--dev`,
  Helm pointer, MCP against an already-running instance, install logging,
  troubleshooting. Start here when GSF is not up yet or something failed
  during bring-up.
- [`skills/gsf-ontology/`](./skills/gsf-ontology/) — inspect and **modify**
  glossary terms, SQL attributes, lineage, model YAML import/export,
  compilation. MCP cannot write; this skill covers the write path.
- [`skills/gsf-agent/`](./skills/gsf-agent/) — how an agent calls GSF
  (MCP first, REST fallback), discovery through the semantic layer, auth
  contracts for AI-Q / SSO / API tokens. Start here when GSF is already
  deployed.

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
- a `components.d/gsf.yml` entry in **nvidia/skills** registering:

```yaml
name: GSF
repo: NVIDIA/GSF
skills:
  - path: skills/gsf-install/
    catalog_dir: gsf-install
  - path: skills/gsf-ontology/
    catalog_dir: gsf-ontology
  - path: skills/gsf-agent/
    catalog_dir: gsf-agent
```

Until that lands, agents use the copies in this checkout. Skills must stay
free of staging hostnames, cluster names, Vault paths, and credentials —
the catalog is public even when GSF itself is access-restricted.
