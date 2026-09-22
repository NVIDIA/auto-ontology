#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

INFRA_SERVICES="postgres pgadmin ingestion-service"
GSF_SERVICES="gsf gsf-frontend"

resolve_infra_services() {
	local services=""
	for svc in $INFRA_SERVICES; do
		services="$services $svc"
	done
	echo "${services# }"
}

# Tear down any previously-running GSF app containers so the chosen mode
# starts from a clean slate (infra containers are left alone).
echo "Stopping existing GSF containers (if any): $GSF_SERVICES"
docker compose rm -sf $GSF_SERVICES >/dev/null 2>&1 || true

if [[ " $* " =~ \ --dev\  ]]; then
	SERVICES=$(resolve_infra_services)
	echo "Starting infrastructure: $SERVICES"
	docker compose up -d --build $SERVICES

	echo ""
	echo "Services:"
	echo "  Postgres:      http://localhost:5432"
	echo "  pgAdmin:       http://localhost:5050"
	echo "  GSF ingestion: http://localhost:3002"

	echo ""
	echo "Start GSF locally:"
	echo "  uv run python -m gsf.server"
	exit 0
fi

if [[ " $* " =~ \ --ds\  ]]; then
	SERVICES=$(resolve_infra_services)
	# Frontend container reaches the host-side backend via host.docker.internal.
	# Override PYTHON_API_URL for both the build arg (baked into Next.js
	# rewrites) and the runtime env.
	export PYTHON_API_URL="http://host.docker.internal:3001"
	echo "Starting infrastructure + gsf-frontend: $SERVICES gsf-frontend"
	echo "(backend will run locally on the host)"
	# --no-deps skips the gsf dependency declared on gsf-frontend.
	docker compose up -d --build --no-deps $SERVICES gsf-frontend

	echo ""
	echo "Services:"
	echo "  Postgres:      http://localhost:5432"
	echo "  pgAdmin:       http://localhost:5050"
	echo "  GSF frontend:  http://localhost:3000"
	echo "  GSF ingestion: http://localhost:3002"

	echo ""
	echo "Start the backend locally:"
	echo "  uv sync"
	echo "  uv run python -m gsf.server"
	echo "  (or use the \"Debug API\" config in .vscode/launch.json)"
	exit 0
fi

# Default: full docker stack (infra + gsf + gsf-frontend)
SERVICES="$(resolve_infra_services) $GSF_SERVICES"
echo "Starting full stack: $SERVICES"
docker compose up -d --build $SERVICES

echo ""
echo "Services:"
echo "  Postgres:      http://localhost:5432"
echo "  pgAdmin:       http://localhost:5050"
echo "  GSF frontend:  http://localhost:3000"
echo "  GSF backend:   http://localhost:3001"
echo "  GSF ingestion: http://localhost:3002"
