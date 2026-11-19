<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Contributing to Auto Ontology

We welcome contributions: bug fixes, connectors, documentation, and examples.
This guide covers how to propose a change, set up a development environment,
and get a pull request merged.

Report security vulnerabilities through [SECURITY.md](SECURITY.md), not GitHub
issues.

## Before you start

- **Bugs:** search the existing issues, then open one with steps to reproduce,
  the commit or version, the connector, and the model you used.
- **Fixes for an open issue:** comment on the issue to say you're working on
  it, so work isn't duplicated. Issues labeled `good first issue` are a good
  place to start.
- **Connectors and integrations:** open an
  [integration request](https://github.com/NVIDIA/auto-ontology/issues/new?template=integration.yml)
  and agree on the approach before writing code.
- **Larger changes:** open an issue that describes the problem and the proposed
  design before sending a pull request.

## Development setup

You need Python 3.12 or 3.13, [uv](https://docs.astral.sh/uv/), Docker, and Node.js 22
with [pnpm](https://pnpm.io/) for the frontend.

```bash
uv sync --extra kumo                   # Python dependencies, including Kumo
(cd frontend && pnpm install)          # frontend dependencies
```

The `kumo` extra ships wheels for Linux and Apple silicon only. On an Intel Mac,
run `uv sync` without it; the Kumo tests then fail locally and run in CI.

To run the whole stack locally, follow the [Quickstart](README.md#quickstart)
in the README.

## Checks before a pull request

These are the checks CI runs.

**Python lint and format:**

```bash
uv run ruff check auto_ontology alembic dev_tools
uv run ruff format --check auto_ontology alembic dev_tools
```

**Python tests.** Many need Postgres with pgvector and the fixture databases.
Use a dedicated container on its own port: the fixture seed resets every
catalog and semantic row in the database it connects to.

```bash
docker run -d --name auto-ontology-test -p 127.0.0.1:5433:5432 \
  -e POSTGRES_USER=auto_ontology -e POSTGRES_PASSWORD=auto_ontology \
  -e POSTGRES_DB=auto_ontology pgvector/pgvector:pg17 &&
until docker exec auto-ontology-test pg_isready -U auto_ontology >/dev/null 2>&1; do sleep 1; done &&
export POSTGRES_HOST=localhost POSTGRES_PORT=5433 POSTGRES_USER=auto_ontology \
  POSTGRES_PASSWORD=auto_ontology POSTGRES_DATABASE=auto_ontology &&
uv run python -m dev_tools.fixtures.seed_fixtures &&
uv run alembic upgrade head &&
CONNECTION_STRINGS="postgresql://auto_ontology:auto_ontology@localhost:5433/pagila,sqlite:///$PWD/dev_tools/fixtures/sql/chinook.sqlite" \
  uv run python -m dev_tools.fixtures.seed_graph_fixture --reset &&
uv run pytest auto_ontology dev_tools -q
```

**MCP server:** `uv run ruff check mcp && uv run ruff format --check mcp && (cd mcp && uv run pytest -q)`

**Frontend:** `(cd frontend && pnpm lint && pnpm format:check && pnpm build)`

**When you change an API model or route,** regenerate the OpenAPI specs; CI
fails when the committed specs drift:

```bash
uv run python -m dev_tools.generate_backend_openapi
(cd frontend && pnpm openapi)
```

**When you change the database schema** (`auto_ontology/dal/schema.py`), add an
Alembic migration with `uv run alembic revision --autogenerate -m "describe the
change"`. CI runs `alembic check`.

## Pull requests

1. Fork the repository and create a branch from `main`.
2. Keep each pull request to one logical change, and link the issue it fixes
   (`Fixes #123`).
3. Add or update tests for the behavior you change.
4. New files must include the appropriate copyright and license header; see
   [Copyright and license headers](#copyright-and-license-headers).
5. Sign off every commit; see [Signing your work](#signing-your-work).
6. Open the pull request, and describe what changed and how you tested it.

After you open it:

- **CI:** a maintainer approves the workflows for first-time contributors. Until
  then, checks show as waiting.
- **Automated review:** CodeRabbit reviews every pull request. Address each
  comment, or reply with your reasoning. Every review thread must be resolved
  before merge.
- **Approval:** a code owner (see `.github/CODEOWNERS`) must approve. Commits
  pushed after an approval can require a fresh one.
- **Skills:** changes under `skills/` are re-signed by a maintainer
  (`/nvskills-ci`) before merge.

## Copyright and license headers

NVIDIA-authored files must use the approved NVIDIA copyright and license header.
For external contributions, retain or add attribution for the actual copyright
owner(s); do not add NVIDIA’s copyright notice to work NVIDIA does not own. DCO
sign-off does not transfer copyright ownership.

## Signing your work

- We require that all contributors "sign-off" on their commits. This certifies
  that the contribution is your original work, or you have rights to submit it
  under the same license, or a compatible license.

  - Any contribution which contains commits that are not Signed-Off will not be
    accepted.

- To sign off on a commit you simply use the `--signoff` (or `-s`) option when
  committing your changes:

  ```bash
  $ git commit -s -m "Add cool feature."
  ```

  This will append the following to your commit message:

  ```
  Signed-off-by: Your Name <your@email.com>
  ```

- Full text of the DCO:

  ```
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    1 Letterman Drive
    Suite D4700
    San Francisco, CA, 94129

    Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
  ```

  ```
    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or

    (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or

    (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.

    (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
  ```

## License

By contributing, you agree that your contributions are licensed under the
[Apache License 2.0](LICENSE).
