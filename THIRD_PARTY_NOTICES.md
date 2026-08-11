# Third-Party Notices

Generative Semantic Fabric (GSF) — Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.

This product includes the third-party open-source software listed below.
The complete text of each license referenced here is reproduced in the
`licenses/` folder of the corresponding upstream project, or is available
at the URL provided. Inclusion of an OSS component on this list does not
imply that the component is statically or dynamically linked into every
build artifact — see `pyproject.toml`, `frontend/package.json`, and the
`Dockerfile`s for actual dependency graphs.

---

## Backend (Python — `pyproject.toml`)

| Component | Version | License | Project URL |
| --- | --- | --- | --- |
| FastAPI | >=0.115.0 | MIT | https://github.com/fastapi/fastapi |
| NeMo Retriever | upstream | Apache-2.0 | https://github.com/NVIDIA/NeMo-Retriever |
| Neo4j Python Driver | >=6.1.0 | Apache-2.0 | https://github.com/neo4j/neo4j-python-driver |
| pandas | >=2.0,<3 | BSD-3-Clause | https://github.com/pandas-dev/pandas |
| psycopg2-binary | >=2.9.11 | LGPL-3.0-or-later (with OpenSSL exception) | https://github.com/psycopg/psycopg2 |
| python-dotenv | >=1.2.2 | BSD-3-Clause | https://github.com/theskumar/python-dotenv |
| PyTorch (torch) | ~=2.9.1 | BSD-3-Clause | https://github.com/pytorch/pytorch |
| torchvision | >=0.24,<0.25 | BSD-3-Clause | https://github.com/pytorch/vision |
| Uvicorn | >=0.32.0 | BSD-3-Clause | https://github.com/encode/uvicorn |

### Backend development dependencies (not redistributed)

| Component | Version | License | Project URL |
| --- | --- | --- | --- |
| HTTPX | >=0.28.1 | BSD-3-Clause | https://github.com/encode/httpx |
| pytest | >=9.0.3 | MIT | https://github.com/pytest-dev/pytest |
| Ruff | >=0.15.9 | MIT | https://github.com/astral-sh/ruff |

### Sample databases (test fixtures, not redistributed in any build artifact)

Vendored under `dev_tools/sql/` and used only to seed local development and
test databases. See `dev_tools/sql/README.md` for provenance and for the
modifications made to Pagila.

| Component | Version | License | Project URL |
| --- | --- | --- | --- |
| Pagila | pagila-v3.1.0 (trimmed; see `dev_tools/sql/README.md`) | MIT | https://github.com/devrimgunduz/pagila |
| Chinook Database | master (unmodified) | MIT | https://github.com/lerocha/chinook-database |

Pagila is Copyright (c) Devrim Gündüz. Chinook is Copyright (c) 2008-2024 Luis
Rocha. Both are distributed under the MIT licence; the full licence text
accompanies each upstream project at the URLs above.

---

## Frontend (Node.js — `frontend/package.json`)

### Runtime dependencies

| Component | Version | License | Project URL |
| --- | --- | --- | --- |
| @nvidia/foundations-react-core | ^0.604.1 | Proprietary (NVIDIA internal) | https://gitlab-master.nvidia.com/maas/kaizen-ui/foundations-react-core |
| @prisma/adapter-pg | ^7.7.0 | Apache-2.0 | https://github.com/prisma/prisma |
| @prisma/client | ^7.7.0 | Apache-2.0 | https://github.com/prisma/prisma |
| axios | ^1.14.0 | MIT | https://github.com/axios/axios |
| dotenv | ^17.4.2 | BSD-2-Clause | https://github.com/motdotla/dotenv |
| Next.js (next) | 16.2.2 | MIT | https://github.com/vercel/next.js |
| node-postgres (pg) | ^8.20.0 | MIT | https://github.com/brianc/node-postgres |
| React (react) | 19.2.4 | MIT | https://github.com/facebook/react |
| React DOM (react-dom) | 19.2.4 | MIT | https://github.com/facebook/react |

### Frontend development / build dependencies

| Component | Version | License | Project URL |
| --- | --- | --- | --- |
| @svgr/webpack | ^8.1.0 | MIT | https://github.com/gregberge/svgr |
| @tailwindcss/postcss | ^4 | MIT | https://github.com/tailwindlabs/tailwindcss |
| @types/node | ^20 | MIT | https://github.com/DefinitelyTyped/DefinitelyTyped |
| @types/pg | ^8.20.0 | MIT | https://github.com/DefinitelyTyped/DefinitelyTyped |
| @types/react | ^19 | MIT | https://github.com/DefinitelyTyped/DefinitelyTyped |
| @types/react-dom | ^19 | MIT | https://github.com/DefinitelyTyped/DefinitelyTyped |
| ESLint | ^9 | MIT | https://github.com/eslint/eslint |
| eslint-config-next | 16.2.2 | MIT | https://github.com/vercel/next.js |
| Prettier | ^3.8.1 | MIT | https://github.com/prettier/prettier |
| Prisma CLI (prisma) | ^7.7.0 | Apache-2.0 | https://github.com/prisma/prisma |
| Tailwind CSS | ^4 | MIT | https://github.com/tailwindlabs/tailwindcss |
| TypeScript | ^5 | Apache-2.0 | https://github.com/microsoft/TypeScript |
| Vite | ^8.0.4 | MIT | https://github.com/vitejs/vite |
| Vitest | ^4.1.2 | MIT | https://github.com/vitest-dev/vitest |

---

## License texts

- **Apache License 2.0** — see [`LICENSE`](./LICENSE) in this repository, or
  https://www.apache.org/licenses/LICENSE-2.0.
- **MIT License** — https://opensource.org/license/mit/
- **BSD 2-Clause "Simplified" License** — https://opensource.org/license/bsd-2-clause/
- **BSD 3-Clause "New" or "Revised" License** — https://opensource.org/license/bsd-3-clause/
- **GNU Lesser General Public License v3.0 or later (LGPL-3.0-or-later)** —
  https://www.gnu.org/licenses/lgpl-3.0.html

The Apache 2.0 license under which GSF is distributed is fully compatible
with each of the licenses listed above when those components are used as
unmodified upstream dependencies. If you redistribute any modified copy of
a third-party component, you must preserve and follow the terms of that
component's original license in addition to the terms of `LICENSE`.

---

## Updating this file

When adding, removing, or upgrading a runtime dependency in
`pyproject.toml` or `frontend/package.json`, update the corresponding row
above in the same change.
