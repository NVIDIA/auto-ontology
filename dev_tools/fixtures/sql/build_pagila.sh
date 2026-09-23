#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Regenerate dev_tools/fixtures/sql/pagila.sql from upstream Pagila.
#
# You should not need to run this — pagila.sql is committed. It exists so the
# fixture is reproducible rather than a mystery blob, and so bumping the
# upstream pin is a one-command change. See README.md for why each flag is here.
#
# Requires Docker. Uses a throwaway container so it never touches the project's
# compose stack or its volumes.
#
# Usage:  ./dev_tools/fixtures/sql/build_pagila.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"; docker rm -f auto-ontology-pagila-build >/dev/null 2>&1 || true' EXIT

# Newest tag that loads on PostgreSQL 17 unmodified. master targets PG18
# (uuidv7() defaults, VIRTUAL generated columns) and will not load here.
PAGILA_REF="pagila-v3.1.0"
RAW="https://raw.githubusercontent.com/devrimgunduz/pagila/${PAGILA_REF}"

# Rentals kept; payments are filtered to match. Everything else is untouched.
KEEP_RENTALS=600

echo "== fetching upstream ${PAGILA_REF} =="
curl -fsS --max-time 180 -o "$WORK/schema.sql" "$RAW/pagila-schema.sql"
curl -fsS --max-time 300 -o "$WORK/data.sql" "$RAW/pagila-data.sql"

echo "== starting throwaway postgres =="
docker run --rm -d --name auto-ontology-pagila-build \
  -e POSTGRES_PASSWORD=fixture -p 55433:5432 postgres:17 >/dev/null

PSQL="docker exec -i -e PGPASSWORD=fixture auto-ontology-pagila-build psql -U postgres -v ON_ERROR_STOP=1"
for _ in $(seq 1 60); do
  if docker exec auto-ontology-pagila-build pg_isready -U postgres >/dev/null 2>&1; then break; fi
  sleep 1
done

echo "== loading =="
$PSQL -d postgres -q -c "CREATE DATABASE pagila_build;"
$PSQL -d pagila_build -q < "$WORK/schema.sql"
$PSQL -d pagila_build -q < "$WORK/data.sql"

echo "== trimming (payment before rental, so the FK never breaks) =="
$PSQL -d pagila_build -q -c "DELETE FROM payment WHERE rental_id IS NULL OR rental_id > ${KEEP_RENTALS};"
$PSQL -d pagila_build -q -c "DELETE FROM rental  WHERE rental_id > ${KEEP_RENTALS};"
$PSQL -d pagila_build -q -c "REFRESH MATERIALIZED VIEW rental_by_category;"

echo "== checking referential integrity =="
$PSQL -d pagila_build -At -c \
  "SELECT 'orphan_payments=' || count(*) FROM payment p
     LEFT JOIN rental r ON r.rental_id = p.rental_id WHERE r.rental_id IS NULL;"
$PSQL -d pagila_build -At -c \
  "SELECT 'orphan_rentals=' || count(*) FROM rental r
     LEFT JOIN inventory i ON i.inventory_id = r.inventory_id WHERE i.inventory_id IS NULL;"

# --inserts: the seed script executes this through psycopg, which cannot run
#            COPY ... FROM stdin.
# --rows-per-insert: a fraction of the size of one INSERT per row.
echo "== dumping =="
docker exec -e PGPASSWORD=fixture auto-ontology-pagila-build pg_dump -U postgres \
  --no-owner --no-privileges --inserts --rows-per-insert=200 \
  pagila_build > "$WORK/dump.sql"

# pg_dump 17.6+ emits \restrict / \unrestrict — psql meta-commands, not SQL.
grep -v '^\\restrict' "$WORK/dump.sql" \
  | grep -v '^\\unrestrict' \
  | sed 's/pagila_build/pagila/g' > "$HERE/pagila.sql"

echo "== done =="
ls -la "$HERE/pagila.sql"
