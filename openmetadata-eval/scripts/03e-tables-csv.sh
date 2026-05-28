#!/usr/bin/env bash
# Check 3e: list tables with columns+description for a database, save raw
# JSON, then flatten to CSV.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .env

DB_FQN="snowflake_eval.SNOWFLAKE_SAMPLE_DATA"
EVID=evidence

# 1. Raw GET — task asked for ?fields=columns,description&database=<fqn>&limit=50
curl -sS -G \
  --data-urlencode "database=$DB_FQN" \
  --data-urlencode "fields=columns,description" \
  --data-urlencode "limit=50" \
  -H "Authorization: Bearer $OM_TOKEN" \
  "$OM_API/v1/tables" \
  > "$EVID/03e-tables.json"

TOTAL=$(jq '.paging.total' "$EVID/03e-tables.json")
RETURNED=$(jq '.data | length' "$EVID/03e-tables.json")
echo "tables total=$TOTAL returned=$RETURNED"

# 2. Flatten to CSV: db, schema, table, column, dataType, description
jq -r '
  ["db","schema","table","column","dataType","description"],
  ( .data[] as $t
    | $t.fullyQualifiedName | split(".") as $parts
    | $t.columns[]
    | [
        $parts[1],
        $parts[2],
        $parts[3],
        .name,
        .dataType,
        ((.description // "") | gsub("[\r\n]+";" "))
      ]
  )
  | @csv
' "$EVID/03e-tables.json" > "$EVID/03e-tables.csv"

echo "csv rows: $(($(wc -l < $EVID/03e-tables.csv) - 1))"
echo "preview:"
head -5 "$EVID/03e-tables.csv"
echo "..."
tail -3 "$EVID/03e-tables.csv"
