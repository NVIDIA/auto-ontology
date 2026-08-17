#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Guarded Prisma schema sync for the `frontend-migrate` Helm hook.
#
# Runs from /app/migrate, where the image ships the Prisma CLI plus
# prisma.config.ts (datasource url) and prisma/schema.prisma.
#
# Why the guard: conversation_analytics gained a REQUIRED user_id column.
# `prisma db push` cannot add a NOT NULL column to a table that already holds
# rows, so a DB seeded before the column existed makes the push fail with
# "Added the required column `user_id` ... without a default value". Those
# pre-feature rows have no user to attribute them to, so we drop them once.
#
# The cleanup is idempotent: it only fires when conversation_analytics exists
# WITHOUT user_id, so it is a no-op on fresh installs (table absent) and on
# every deploy after the column has been added. It touches only the `public`
# schema, matching the scope `db push` operates on.
set -eu

# Both subcommands read the schema path and datasource url from
# prisma.config.ts (shipped alongside this script in /app/migrate). Prisma 7's
# `db execute` has no --schema flag; it takes the datasource from the config.
PRISMA="node node_modules/prisma/build/index.js"

# `db push` does not create the schema it targets, so this has to exist first
# or the very first push fails with "schema \"frontend\" does not exist".
echo "migrate: ensuring the frontend schema exists..."
$PRISMA db execute --stdin <<'SQL'
CREATE SCHEMA IF NOT EXISTS frontend;
SQL

echo "migrate: checking conversation_analytics for the pre-user_id schema..."
$PRISMA db execute --stdin <<'SQL'
DO $$
DECLARE
  stale_rows bigint;
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.tables
    WHERE table_schema = 'frontend' AND table_name = 'conversation_analytics'
  ) AND NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'frontend'
      AND table_name = 'conversation_analytics'
      AND column_name = 'user_id'
  ) THEN
    SELECT count(*) INTO stale_rows FROM frontend.conversation_analytics;
    RAISE NOTICE 'conversation_analytics is missing user_id; truncating % pre-feature row(s) so the required column can be added', stale_rows;
    TRUNCATE TABLE frontend.conversation_analytics;
  END IF;
END $$;
SQL

echo "migrate: running prisma db push..."
exec $PRISMA db push
