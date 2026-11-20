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
(`.env` for Docker Compose, `--set connectionStrings=...` for Helm). A comma
separates two URLs only where the next one starts, so the commas inside a
MongoDB host list stay part of their URL. It carries no schema or table
filters. `GET /api/connections/source` reports whether it is set.

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
| MongoDB, Amazon DocumentDB | `mongodb://`, `mongodb+srv://` | no |

The registry lives in
[`auto_ontology/connectors/registry.py`](../auto_ontology/connectors/registry.py).

## MongoDB and Amazon DocumentDB

MongoDB has no SQL engine, so the connector runs each statement on a private,
in-process DuckDB database loaded with the documents that statement reads. It
reports the `duckdb` dialect, and generated SQL is DuckDB SQL.

- **Tables and columns.** Each collection is a table in schema `main`, and so
  is each view on servers that have views. Each top-level field is a column,
  typed from a `$sample` of the collection; a BSON type held by fewer than 1%
  of the sampled values does not change the column's type. Embedded documents
  and arrays are `JSON` columns, read with `->` and `->>`. `_id` is the
  primary key. MongoDB declares no foreign keys, so relationships come from
  semantic compilation.
- **Reading.** Only the fields a statement names are fetched. For a statement
  over one collection, simple `WHERE` predicates (`=` and `IN`, ranges on
  numbers and dates, `IS NOT NULL`) and a bare `LIMIT` are applied by the
  server. Everything else is applied by DuckDB, which runs with external
  access disabled, so generated SQL cannot read files or load extensions.
- **Limits.** One statement reads at most `max_documents` documents from one
  collection. Past that it fails rather than answering from part of the
  collection; narrow it with a filter the server can apply, or raise the
  limit. A value whose BSON type differs from its column's type reads as
  NULL, and so does a NaN.

Put the database to ingest in the URL path. The connector reads three options
of its own and removes them before the URL reaches PyMongo; every other
option is PyMongo's.

| Option          | Default | Meaning |
| --------------- | ------- | ------- |
| `sample_size`   | 1000    | Documents sampled per collection to type its columns |
| `max_documents` | 100000  | Documents one statement may read from one collection |
| `tls_ca_data`   | —       | Base64 of a PEM CA bundle, used instead of a `tlsCAFile` path |

```text
mongodb://reader:<password>@mongo.example.com:27017/shop?tls=true&authSource=admin
mongodb://reader:<password>@mongo-1:27017,mongo-2:27017/shop?tls=true&replicaSet=rs0&authSource=admin
mongodb+srv://reader:<password>@cluster0.example.mongodb.net/shop
```

PyMongo connects without TLS unless the URL asks for it (`mongodb+srv://`
turns it on by default), so keep `tls=true` for any server reached over a
network you do not control.

For Amazon DocumentDB, use the cluster endpoint with the options AWS gives
for drivers: TLS with the Amazon RDS CA bundle, `replicaSet=rs0`,
`readPreference=secondaryPreferred` and `retryWrites=false`. DocumentDB
authenticates users against `admin` whichever database the path names.

```text
mongodb://reader:<password>@docdb.cluster-xxxx.us-east-1.docdb.amazonaws.com:27017/shop?tls=true&tlsCAFile=/certs/global-bundle.pem&replicaSet=rs0&readPreference=secondaryPreferred&retryWrites=false
```

Where the CA file cannot be mounted, as with the Helm chart, pass your
region's bundle inline instead of `tlsCAFile`, as URL-safe base64:

```bash
curl -sO https://truststore.pki.rds.amazonaws.com/us-east-1/us-east-1-bundle.pem
python3 -c "import base64, sys; print(base64.urlsafe_b64encode(open(sys.argv[1], 'rb').read()).decode())" us-east-1-bundle.pem
# ...&tls_ca_data=<output>
```

A regional bundle is about 5 KB. The global bundle is about 175 KB, which
is over Linux's 128 KiB limit for one environment variable.

The servers supported are the ones the bundled PyMongo supports: MongoDB 4.4
and later, and Amazon DocumentDB 5.0 and 8.0. DocumentDB 4.0 speaks the
MongoDB 4.0 wire protocol, which PyMongo dropped in 4.14. DocumentDB elastic
clusters do not accept `maxTimeMS`; there the connector stops a slow fetch
with a client-side deadline instead.

## Vault

Vault is off by default. It turns on only when all four of these are set:

- `VAULT_ADDR`
- `VAULT_NAMESPACE`
- `VAULT_ROLE_ID`
- `VAULT_SECRET_ID`

`VAULT_AUTH_MOUNT` and `VAULT_KV_MOUNT` are optional overrides. A partial
configuration is ignored with a warning, and credentials fall back to plaintext
storage in Postgres.
