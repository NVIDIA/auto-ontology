#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

INFRA_SERVICES="postgres pgadmin ingestion-service"
AUTO_ONTOLOGY_SERVICES="auto-ontology auto-ontology-frontend"

# A variable from the environment, else its last assignment in .env.
env_value() {
	local name="$1"
	if [[ -n "${!name:-}" ]]; then
		echo "${!name}"
	elif [[ -f .env ]]; then
		# A quoted value is what is between its quotes, so a # inside stays
		# and a comment after the closing quote goes. An unquoted value loses
		# a trailing comment and surrounding whitespace.
		sed -n "s/^[[:space:]]*${name}=//p" .env | tail -n 1 |
			sed -E -e 's/^[[:space:]]*"([^"]*)".*$/\1/' -e t \
				-e "s/^[[:space:]]*'([^']*)'.*\$/\1/" -e t \
				-e 's/[[:space:]]+#.*$//' -e 's/^[[:space:]]+//' -e 's/[[:space:]]+$//'
	fi
}

# Elasticsearch vector store. With VDB_BACKEND=elasticsearch, start the local
# single-node cluster unless ES_URL points somewhere else (Elastic Cloud,
# Serverless, another local cluster): ES_URL unset or http://localhost:<ES_PORT>
# means the bundled one. Containers reach it by service name, so ES_DOCKER_URL
# overrides their ES_URL (see docker-compose.yml).
COMPOSE_PROFILES_ARGS=""
ES_LINE=""
VDB_BACKEND_VALUE="$(env_value VDB_BACKEND | tr '[:upper:]' '[:lower:]')"
if [[ "$VDB_BACKEND_VALUE" == "elasticsearch" ]]; then
	ES_PORT="$(env_value ES_PORT)"
	ES_PORT="${ES_PORT:-9200}"
	export ES_PORT
	ES_URL_VALUE="$(env_value ES_URL)"
	case "${ES_URL_VALUE%/}" in
	"" | "http://localhost:${ES_PORT}" | "http://127.0.0.1:${ES_PORT}")
		INFRA_SERVICES="elasticsearch $INFRA_SERVICES"
		COMPOSE_PROFILES_ARGS="--profile elasticsearch"
		export ES_DOCKER_URL="http://elasticsearch:9200"
		ES_LINE="  Elasticsearch: http://localhost:${ES_PORT}"
		if [[ -z "$ES_URL_VALUE" ]]; then
			echo "NOTE: add ES_URL=http://localhost:${ES_PORT} to .env so a backend run on the host finds it."
		fi
		;;
	*)
		echo "VDB_BACKEND=elasticsearch: using the cluster at $ES_URL_VALUE (not starting one)."
		# A cluster on the host is host.docker.internal from inside a container.
		ES_DOCKER_URL="$(echo "$ES_URL_VALUE" | sed -E 's#//(localhost|127\.0\.0\.1)([:/]|$)#//host.docker.internal\2#')"
		[[ "$ES_DOCKER_URL" != "$ES_URL_VALUE" ]] && export ES_DOCKER_URL
		ES_LINE="  Elasticsearch: $ES_URL_VALUE (external)"
		;;
	esac
fi

resolve_infra_services() {
	local services=""
	for svc in $INFRA_SERVICES; do
		services="$services $svc"
	done
	echo "${services# }"
}

# Tear down any previously-running Auto Ontology app containers so the chosen mode
# starts from a clean slate (infra containers are left alone).
echo "Stopping existing Auto Ontology containers (if any): $AUTO_ONTOLOGY_SERVICES"
docker compose $COMPOSE_PROFILES_ARGS rm -sf $AUTO_ONTOLOGY_SERVICES >/dev/null 2>&1 || true

if [[ " $* " =~ \ --dev\  ]]; then
	SERVICES=$(resolve_infra_services)
	echo "Starting infrastructure: $SERVICES"
	docker compose $COMPOSE_PROFILES_ARGS up -d --build $SERVICES

	echo ""
	echo "Services:"
	echo "  Postgres:      http://localhost:5432"
	[[ -n "$ES_LINE" ]] && echo "$ES_LINE"
	echo "  pgAdmin:       http://localhost:5050"
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
	docker compose $COMPOSE_PROFILES_ARGS up -d --build --no-deps $SERVICES auto-ontology-frontend

	echo ""
	echo "Services:"
	echo "  Postgres:      http://localhost:5432"
	[[ -n "$ES_LINE" ]] && echo "$ES_LINE"
	echo "  pgAdmin:       http://localhost:5050"
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
docker compose $COMPOSE_PROFILES_ARGS up -d --build $SERVICES

echo ""
echo "Services:"
echo "  Postgres:      http://localhost:5432"
[[ -n "$ES_LINE" ]] && echo "$ES_LINE"
echo "  pgAdmin:       http://localhost:5050"
echo "  Auto Ontology frontend:  http://localhost:3000"
echo "  Auto Ontology backend:   http://localhost:3001"
echo "  Auto Ontology ingestion: http://localhost:3002"
