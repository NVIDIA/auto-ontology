#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

INFRA_SERVICES="postgres pgadmin ingestion-service"
AUTO_ONTOLOGY_SERVICES="auto-ontology auto-ontology-frontend"

# The vector store backend, from the shell or else from .env, in lower case.
# Reads .env the way python-dotenv does for this value: a " #" starts a comment.
vdb_backend() {
	local value="${AUTO_ONTOLOGY_VDB_BACKEND:-}"
	if [[ -z "$value" && -f .env ]]; then
		value=$(sed -n 's/^[[:space:]]*\(export[[:space:]]\{1,\}\)\{0,1\}AUTO_ONTOLOGY_VDB_BACKEND[[:space:]]*=//p' .env | tail -n 1)
		value="${value%%[[:space:]]#*}"
	fi
	value="${value//\"/}"
	value="${value//\'/}"
	echo "$value" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]'
}

resolve_infra_services() {
	local services=""
	for svc in $INFRA_SERVICES; do
		services="$services $svc"
	done
	# The qdrant service sits behind a Compose profile. Naming it here starts it.
	if [[ "$(vdb_backend)" == "qdrant" ]]; then
		services="$services qdrant"
	fi
	echo "${services# }"
}

print_qdrant_url() {
	if [[ "$(vdb_backend)" == "qdrant" ]]; then
		echo "  Qdrant:        http://localhost:6333/dashboard"
	fi
}

# Tear down any previously-running Auto Ontology app containers so the chosen mode
# starts from a clean slate (infra containers are left alone).
echo "Stopping existing Auto Ontology containers (if any): $AUTO_ONTOLOGY_SERVICES"
docker compose rm -sf $AUTO_ONTOLOGY_SERVICES >/dev/null 2>&1 || true

if [[ " $* " =~ \ --dev\  ]]; then
	SERVICES=$(resolve_infra_services)
	echo "Starting infrastructure: $SERVICES"
	docker compose up -d --build $SERVICES

	echo ""
	echo "Services:"
	echo "  Postgres:      http://localhost:5432"
	echo "  pgAdmin:       http://localhost:5050"
	print_qdrant_url
	echo "  Auto Ontology ingestion: http://localhost:3002"

	echo ""
	echo "Start Auto Ontology locally:"
	echo "  uv run python -m auto_ontology.server"
	exit 0
fi

if [[ " $* " =~ \ --ds\  ]]; then
	SERVICES=$(resolve_infra_services)
	# Frontend container reaches the host-side backend via host.docker.internal.
	# Override PYTHON_API_URL for both the build arg (baked into Next.js
	# rewrites) and the runtime env.
	export PYTHON_API_URL="http://host.docker.internal:3001"
	echo "Starting infrastructure + auto-ontology-frontend: $SERVICES auto-ontology-frontend"
	echo "(backend will run locally on the host)"
	# --no-deps skips the auto_ontology dependency declared on auto-ontology-frontend.
	docker compose up -d --build --no-deps $SERVICES auto-ontology-frontend

	echo ""
	echo "Services:"
	echo "  Postgres:      http://localhost:5432"
	echo "  pgAdmin:       http://localhost:5050"
	print_qdrant_url
	echo "  Auto Ontology frontend:  http://localhost:3000"
	echo "  Auto Ontology ingestion: http://localhost:3002"

	echo ""
	echo "Start the backend locally:"
	echo "  uv sync"
	echo "  uv run python -m auto_ontology.server"
	echo "  (or use the \"Debug API\" config in .vscode/launch.json)"
	exit 0
fi

# Default: full docker stack (infra + auto_ontology + auto-ontology-frontend)
SERVICES="$(resolve_infra_services) $AUTO_ONTOLOGY_SERVICES"
echo "Starting full stack: $SERVICES"
docker compose up -d --build $SERVICES

echo ""
echo "Services:"
echo "  Postgres:      http://localhost:5432"
echo "  pgAdmin:       http://localhost:5050"
print_qdrant_url
echo "  Auto Ontology frontend:  http://localhost:3000"
echo "  Auto Ontology backend:   http://localhost:3001"
echo "  Auto Ontology ingestion: http://localhost:3002"
