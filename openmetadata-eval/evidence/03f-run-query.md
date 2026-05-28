# 3f — Run a query (NOT SUPPORTED)

OpenMetadata is a metadata catalog and observability platform; it is not a query engine. There is no REST endpoint that executes SQL against a connected database — the platform only ingests, stores, and surfaces metadata, lineage, and query *history* that was collected from the source system's own logs.

## Confirmed against the running 1.12.9 server

Probing the obvious URL shapes against our local OM (see `03f-no-run-query-endpoint.txt`):

```
POST /api/v1/queries/execute  -> HTTP 404
POST /api/v1/queries/run      -> HTTP 404
POST /api/v1/sql/execute      -> HTTP 404
POST /api/v1/query/run        -> HTTP 404
```

`/api/v1/queries` is a CRUD collection for cataloging Query *entities* (records of queries that were observed via lineage/usage ingestion). It accepts `GET`, `POST` (register a query record), `PATCH`, and `DELETE` — never "execute".

## Backed by upstream docs

- The 1.11.x docs page *APIs | OpenMetadata Metadata Standard APIs* enumerates the entire resource taxonomy. Under search & query the only verbs are `/api/v1/search/query` (Elasticsearch full-text over metadata) and `/api/v1/search/suggest` (auto-complete). No "execute SQL" verb exists.
- The 1.13.x search reference reiterates that `/api/v1/search/query` searches **metadata** entities, not database rows.
- The Python SDK's *automated SQL lineage* method (`OpenMetadata.add_lineage_by_query(... sql=...)`) *parses* a query string with sqlfluff/sqlglot to build lineage edges. It does not execute the SQL.

## What people actually want when they ask this question

| Asked for | OpenMetadata's answer |
| --- | --- |
| "Run SQL via OM API" | Not supported. Use the source warehouse's driver directly (Snowflake REST, BigQuery client, etc.). |
| "Search for tables / dashboards by keyword" | `GET /api/v1/search/query?q=customer&index=table_search_index` |
| "Show me historical queries that touched table X" | `GET /api/v1/tables/{id}/tableQuery` (per-table) or `GET /api/v1/queries?fields=query,users,queryDate,queryUsedIn` (cross-service). Populated by the *usage* connector — already validated in check 3g. |
| "Get lineage built from a SQL string without ingesting it" | `OpenMetadata.add_lineage_by_query(...)` Python SDK helper. Parses, does not execute. |

## Result

This check is correctly logged as **N/A — feature is not in scope for OpenMetadata**. Any RFP / build-vs-buy comparison should split "catalog queries" (OM does this well, see 3g) from "execute queries" (OM does not — pair with a query layer like Trino, dbt Cloud SQL Run, Hex, etc. if needed).
