#!/usr/bin/env bash
# Check 3g: query history.
#   - GET /v1/queries?fields=query,users,queryDate,queryUsedIn&limit=20 (cross-service)
#   - Per-table queries via /v1/tables/{id}/tableQuery (and via fields=queries on tables)
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .env

AUTH=(-H "Authorization: Bearer $OM_TOKEN")
EVID=evidence

echo "== 1. Top-level queries collection =="
# The literal request from the spec
curl -sS "${AUTH[@]}" -G "$OM_API/v1/queries" \
  --data-urlencode "fields=query,users,queryDate,queryUsedIn" \
  --data-urlencode "limit=20" \
  > "$EVID/03g-queries.json"
echo "paging:"
jq '.paging' "$EVID/03g-queries.json"
echo "first 3 entries (truncated):"
jq '[.data[0:3][] | {fqn: .fullyQualifiedName, queryDate, users: (.users // []), queryUsedIn: (.queryUsedIn // []), query: (.query[0:160] + (if (.query|length) > 160 then "…" else "" end))}]' "$EVID/03g-queries.json"

echo
echo "== 1b. Queries scoped to our snowflake_eval service =="
SERVICE_ID=$(curl -sS "${AUTH[@]}" "$OM_API/v1/services/databaseServices/name/snowflake_eval" | jq -r '.id')
echo "  service id: $SERVICE_ID"
curl -sS "${AUTH[@]}" -G "$OM_API/v1/queries" \
  --data-urlencode "fields=query,users,queryDate,queryUsedIn" \
  --data-urlencode "service=snowflake_eval" \
  --data-urlencode "limit=20" \
  > "$EVID/03g-queries-by-service.json"
jq '.paging' "$EVID/03g-queries-by-service.json"

echo
echo "== 2. Per-table query history (TPCH_SF1.CUSTOMER) =="
TABLE_FQN="snowflake_eval.SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.CUSTOMER"
TABLE_ID=$(curl -sS "${AUTH[@]}" "$OM_API/v1/tables/name/$TABLE_FQN" | jq -r '.id')
echo "  table id: $TABLE_ID"

curl -sS "${AUTH[@]}" "$OM_API/v1/tables/name/$TABLE_FQN?fields=queries,joins,usageSummary" \
  > "$EVID/03g-table-customer-queries.json"
echo "summary:"
jq '{
  fqn: .fullyQualifiedName,
  usage_today: .usageSummary.dailyStats,
  usage_week:  .usageSummary.weeklyStats,
  query_count: (.queries // [] | length),
  first_queries: [(.queries // [])[0:3][] | {queryDate, users: (.users // []), preview: (.query[0:160] + (if (.query|length) > 160 then "…" else "" end))}]
}' "$EVID/03g-table-customer-queries.json"

echo
echo "== 3. Sanity: aggregate query counts across our 32 ingested tables =="
curl -sS "${AUTH[@]}" "$OM_API/v1/tables?database=snowflake_eval.SNOWFLAKE_SAMPLE_DATA&fields=usageSummary&limit=200" \
  | jq '.data
        | map({fqn:.fullyQualifiedName,
               weekly_queries:(.usageSummary.weeklyStats.count // 0),
               weekly_pct:(.usageSummary.weeklyStats.percentileRank // 0)})
        | sort_by(-.weekly_queries)
        | .[0:10]' \
  | tee "$EVID/03g-top-tables.json"
