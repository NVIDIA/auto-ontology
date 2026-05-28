#!/usr/bin/env bash
# Poll the OM server until /api/v1/system/version returns 200 or we time out.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .env

DEADLINE=$(( $(date +%s) + 900 ))
echo "Waiting for ${OM_API}/v1/system/version (timeout: 15 min)..."
while (( $(date +%s) < DEADLINE )); do
  if STATUS=$(curl -s -o /tmp/om-version.json -w '%{http_code}' "${OM_API}/v1/system/version") && [[ "$STATUS" == "200" ]]; then
    echo "OM server is up:"
    cat /tmp/om-version.json
    echo
    exit 0
  fi
  echo "  not ready yet (status=${STATUS:-?}); sleeping 10s"
  sleep 10
done
echo "FATAL: OM server did not become ready within 15 minutes" >&2
exit 1
