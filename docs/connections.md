<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Connections and credential storage

Auto-Ontology reads source databases from two places: connections added in the
UI, and the `CONNECTION_STRINGS` environment variable. Use one per deployment,
because the two code paths order them differently:

- **Answering questions** uses UI-managed connections when any exist, and falls
  back to `CONNECTION_STRINGS`.
- **Ingestion** uses `CONNECTION_STRINGS` whenever it's set, and reads
  UI-managed connections only when it's empty.

## Connections added in the UI

Add connections under **Settings → Connections**, or through the API
(`POST /api/connections/test`, then `POST /api/connections`; permission
`connection:manage`).

Each connection can be scoped with a schema allowlist plus
`table_allow_regex` / `table_deny_regex`, applied at catalog extraction. Deny
wins, and names are unqualified and case-sensitive.

Where the credentials live depends on whether Vault is configured:

- **Without Vault.** The full connection object, password included, is
  serialized to JSON and stored unencrypted on the catalog's database row in
  Postgres.
- **With Vault.** Credentials are written to Vault, keyed by database name, and
  the catalog row stores none. The database name is the lookup key.

## `CONNECTION_STRINGS`

A comma- or newline-separated list of connection URLs supplied at deploy time
(`.env` for Docker Compose, `--set connectionStrings=...` for Helm). It carries
no schema or table filters. `GET /api/connections/source` reports whether it is set.

The URL scheme selects the connector:

| Engine        | Scheme                      | Add in the UI |
| ------------- | --------------------------- | :-----------: |
| PostgreSQL    | `postgresql://`, `postgres://` | yes        |
| MySQL         | `mysql://`                  | yes           |
| Databricks    | `databricks://`             | yes           |
| Snowflake     | `snowflake://`              | yes           |
| ClickHouse    | `clickhouse://`             | yes           |
| Trino         | `trino://`                  | yes           |
| Apache Kyuubi | `kyuubi://`                 | yes           |
| HeavyDB       | `heavydb://`                | yes           |
| DuckDB        | `duckdb://`                 | no            |
| SQLite        | `sqlite://`                 | no            |

The registry lives in
[`auto_ontology/connectors/registry.py`](../auto_ontology/connectors/registry.py).

## Vault

Vault is off by default. It turns on only when all four of these are set:

- `VAULT_ADDR`
- `VAULT_NAMESPACE`
- `VAULT_ROLE_ID`
- `VAULT_SECRET_ID`

`VAULT_AUTH_MOUNT` and `VAULT_KV_MOUNT` are optional overrides. A partial
configuration is ignored with a warning, and credentials fall back to plaintext
storage in Postgres.
