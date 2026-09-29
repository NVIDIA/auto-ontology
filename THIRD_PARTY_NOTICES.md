# Third-Party Notices

Auto Ontology — Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.

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
| Alembic | >=1.19.1 | MIT | https://github.com/sqlalchemy/alembic |
| apache-ossie-nvidia-gsf | git (`converters/nvidia`) | Apache-2.0 | https://github.com/apache/ossie |
| databricks-sql-connector | >=4.4.0 | Apache-2.0 | https://github.com/databricks/databricks-sql-python |
| DuckDB | >=1.5.5 | MIT | https://github.com/duckdb/duckdb-python |
| FastAPI | >=0.141.1 | MIT | https://github.com/fastapi/fastapi |
| HTTPX | >=0.28.1 | BSD-3-Clause | https://github.com/encode/httpx |
| hvac | >=2.4.0 | Apache-2.0 | https://github.com/hvac/hvac |
| importlib-metadata | >=8.7.1 | Apache-2.0 | https://github.com/python/importlib_metadata |
| langchain-openai | >=1.6.0 | MIT | https://github.com/langchain-ai/langchain |
| langchain-postgres | >=0.0.17 | MIT | https://github.com/langchain-ai/langchain-postgres |
| LangGraph | >=1.2.11 | MIT | https://github.com/langchain-ai/langgraph |
| mysql-connector-python | >=26.7.0 | GPL-2.0 with Universal FOSS Exception — see note below | https://github.com/mysql/mysql-connector-python |
| NeMo Retriever | >=26.8.1 | Apache-2.0 | https://github.com/NVIDIA/NeMo-Retriever |
| nvidia-sdfm (`nvidia-sdfm[kumorfm]`) | >=0.2.1 | Apache-2.0 — see note below | https://github.com/NVIDIA/nvidia-sdfm-sdk |
| pandas | >=2.3.3,<3 | BSD-3-Clause | https://github.com/pandas-dev/pandas |
| psycopg (`psycopg[binary]`) | >=3.3.4 | LGPL-3.0-only — see note below | https://github.com/psycopg/psycopg |
| psycopg-pool | >=3.3.1 | LGPL-3.0-only | https://github.com/psycopg/psycopg |
| pure-sasl | >=0.6.2 | MIT | https://github.com/thobbs/pure-sasl |
| Pydantic | >=2.13.5 | MIT | https://github.com/pydantic/pydantic |
| pyheavydb | >=8.0.1.post1 | Apache-2.0 | https://github.com/heavyai/pyheavydb |
| PyHive (`pyhive[hive-pure-sasl]`) | >=0.7.0 | Apache-2.0 | https://github.com/dropbox/PyHive |
| pyjks | >=20.0.0 | MIT | https://github.com/kurtbrose/pyjks |
| python-dotenv | >=1.2.3 | BSD-3-Clause | https://github.com/theskumar/python-dotenv |
| PyYAML | >=6.0.3 | MIT | https://github.com/yaml/pyyaml |
| snowflake-connector-python | >=4.7.2 | Apache-2.0 | https://github.com/snowflakedb/snowflake-connector-python |
| SQLAlchemy | >=2.0.52 | MIT | https://github.com/sqlalchemy/sqlalchemy |
| Thrift | >=0.24.0 | Apache-2.0 | https://github.com/apache/thrift |
| thrift-sasl | >=0.4.3 | Apache-2.0 | https://github.com/cloudera/thrift_sasl |
| trino | >=0.339.0 | Apache-2.0 | https://github.com/trinodb/trino-python-client |
| Uvicorn (`uvicorn[standard]`) | >=0.52.4 | BSD-3-Clause | https://github.com/Kludex/uvicorn |

The `binary` extra installs the `psycopg-binary` wheel, which bundles
prebuilt native libraries rather than linking the system ones. Those carry
their own licenses and are redistributed inside any image built from this
repo: libpq (PostgreSQL License), OpenSSL 3 (Apache-2.0), MIT Kerberos
(MIT-style), and OpenLDAP (OpenLDAP Public License). Building against a
system libpq instead — `psycopg[c]` or plain `psycopg` — avoids bundling
them.

`mysql-connector-python` is licensed under GPL-2.0 with Oracle's Universal
FOSS Exception, which lets it be combined with software under OSI-approved
licenses such as Apache-2.0. It is used unmodified.

`nvidia-sdfm[kumorfm]` resolves to NVIDIA wheels vendored under
`vendor/wheels/`: `nvidia-sdfm` 0.2.1, `sdfm-connectors` 0.3.0 and `kumorfm`
2.29.0, all Apache-2.0. `kumorfm` bundles Mermaid (MIT); its license text
ships in the wheel as `kumorfm/rfm/assets/MERMAID_LICENSE`.

### Backend development dependencies (not redistributed)

| Component | Version | License | Project URL |
| --- | --- | --- | --- |
| HTTPX | >=0.28.1 | BSD-3-Clause | https://github.com/encode/httpx |
| pytest | >=9.1.1 | MIT | https://github.com/pytest-dev/pytest |
| Ruff | >=0.16.5 | MIT | https://github.com/astral-sh/ruff |

### Sample databases (test fixtures, not redistributed in any build artifact)

Vendored under `dev_tools/fixtures/sql/` and used only to seed local development and
test databases. See `dev_tools/fixtures/sql/README.md` for provenance and for the
modifications made to Pagila.

| Component | Version | License | Project URL |
| --- | --- | --- | --- |
| Pagila | pagila-v3.1.0 (trimmed; see `dev_tools/fixtures/sql/README.md`) | MIT | https://github.com/devrimgunduz/pagila |
| Chinook Database | master (unmodified) | MIT | https://github.com/lerocha/chinook-database |

Pagila is Copyright (c) Devrim Gündüz. Chinook is Copyright (c) 2008-2024 Luis
Rocha. Both are distributed under the MIT licence; the full licence text
accompanies each upstream project at the URLs above.

---

## Frontend (Node.js — `frontend/package.json`)

### Runtime dependencies

| Component | Version | License | Project URL |
| --- | --- | --- | --- |
| @better-auth/api-key | 1.7.4 | MIT | https://github.com/better-auth/better-auth |
| @better-auth/core | 1.7.4 | MIT | https://github.com/better-auth/better-auth |
| @better-auth/mcp | 1.7.4 | MIT | https://github.com/better-auth/better-auth |
| @better-auth/oauth-provider | 1.7.4 | MIT | https://github.com/better-auth/better-auth |
| @better-auth/sso | 1.7.4 | MIT | https://github.com/better-auth/better-auth |
| @nvidia/foundations-react-core (Kaizen UI) | ^0.604.1 | Apache-2.0 | https://www.npmjs.com/package/@nvidia/foundations-react-core |
| @prisma/adapter-pg | ^7.7.0 | Apache-2.0 | https://github.com/prisma/prisma |
| @prisma/client | ^7.7.0 | Apache-2.0 | https://github.com/prisma/prisma |
| @sigma/node-border | ^3.0.0 | MIT | https://github.com/jacomyal/sigma.js |
| @tanstack/react-query | ^5.103.2 | MIT | https://github.com/TanStack/query |
| axios | ^1.20.0 | MIT | https://github.com/axios/axios |
| better-auth | 1.7.4 | MIT | https://github.com/better-auth/better-auth |
| d3-force | ^3.0.0 | ISC | https://github.com/d3/d3-force |
| dayjs | ^1.11.21 | MIT | https://github.com/iamkun/dayjs |
| dotenv | ^17.4.2 | BSD-2-Clause | https://github.com/motdotla/dotenv |
| graphology | ^0.26.0 | MIT | https://github.com/graphology/graphology |
| jose | ^6.1.0 | MIT | https://github.com/panva/jose |
| Next.js (next) | 16.3.5 | MIT | https://github.com/vercel/next.js |
| node-postgres (pg) | ^8.20.0 | MIT | https://github.com/brianc/node-postgres |
| React (react) | 19.2.4 | MIT | https://github.com/facebook/react |
| React DOM (react-dom) | 19.2.4 | MIT | https://github.com/facebook/react |
| sigma | ^3.0.3 | MIT | https://github.com/jacomyal/sigma.js |
| zod | ^4.4.3 | MIT | https://github.com/colinhacks/zod |

### Frontend development / build dependencies

| Component | Version | License | Project URL |
| --- | --- | --- | --- |
| @svgr/webpack | ^8.1.0 | MIT | https://github.com/gregberge/svgr |
| @tailwindcss/postcss | ^4 | MIT | https://github.com/tailwindlabs/tailwindcss |
| @types/d3-force | ^3.0.10 | MIT | https://github.com/DefinitelyTyped/DefinitelyTyped |
| @types/node | ^20 | MIT | https://github.com/DefinitelyTyped/DefinitelyTyped |
| @types/pg | ^8.20.0 | MIT | https://github.com/DefinitelyTyped/DefinitelyTyped |
| @types/react | ^19 | MIT | https://github.com/DefinitelyTyped/DefinitelyTyped |
| @types/react-dom | ^19 | MIT | https://github.com/DefinitelyTyped/DefinitelyTyped |
| ESLint | ^9 | MIT | https://github.com/eslint/eslint |
| eslint-config-next | 16.3.5 | MIT | https://github.com/vercel/next.js |
| Prettier | ^3.8.1 | MIT | https://github.com/prettier/prettier |
| Prisma CLI (prisma) | ^7.7.0 | Apache-2.0 | https://github.com/prisma/prisma |
| Tailwind CSS | ^4 | MIT | https://github.com/tailwindlabs/tailwindcss |
| ts-morph | ^28.0.0 | MIT | https://github.com/dsherret/ts-morph |
| tsx | ^4.23.11 | MIT | https://github.com/privatenumber/tsx |
| TypeScript | ^5 | Apache-2.0 | https://github.com/microsoft/TypeScript |
| Vite | ^8.3.0 | MIT | https://github.com/vitejs/vite |
| Vitest | ^4.1.11 | MIT | https://github.com/vitest-dev/vitest |

---

## License texts

- **Apache License 2.0** — see [`LICENSE`](./LICENSE) in this repository, or
  https://www.apache.org/licenses/LICENSE-2.0.
- **MIT License** — https://opensource.org/license/mit/
- **BSD 2-Clause "Simplified" License** — https://opensource.org/license/bsd-2-clause/
- **BSD 3-Clause "New" or "Revised" License** — https://opensource.org/license/bsd-3-clause/
- **GNU Lesser General Public License v3.0 only (LGPL-3.0-only)** —
  https://www.gnu.org/licenses/lgpl-3.0.html
- **GNU General Public License v2.0 (GPL-2.0)** —
  https://www.gnu.org/licenses/old-licenses/gpl-2.0.html; **Universal FOSS
  Exception, Version 1.0** — https://oss.oracle.com/licenses/universal-foss-exception/
- **ISC License** — https://opensource.org/license/isc-license-txt/
- **PostgreSQL License** — https://opensource.org/license/postgresql/
- **OpenLDAP Public License** — https://www.openldap.org/software/release/license.html

The Apache 2.0 license under which Auto Ontology is distributed is fully compatible
with each of the licenses listed above when those components are used as
unmodified upstream dependencies. If you redistribute any modified copy of
a third-party component, you must preserve and follow the terms of that
component's original license in addition to the terms of `LICENSE`.

---

## Updating this file

The tables above list the **direct** dependencies declared in
`pyproject.toml` and `frontend/package.json`, not the full transitive
closure. When adding, removing, or upgrading a direct dependency, update
the corresponding row in the same change. For the resolved closure and its
pinned versions, see `uv.lock` and `frontend/pnpm-lock.yaml`.
