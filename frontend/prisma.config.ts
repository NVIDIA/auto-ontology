// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { config } from 'dotenv';
import { defineConfig } from 'prisma/config';

config({ path: '../.env' });

const user = process.env.POSTGRES_USER ?? 'postgres';
const password = process.env.POSTGRES_PASSWORD ?? '';
const host = process.env.POSTGRES_HOST ?? 'localhost';
const port = process.env.POSTGRES_PORT ?? '5432';
const database = process.env.POSTGRES_DATABASE ?? 'gsf';

export default defineConfig({
	schema: 'prisma/schema.prisma',
	migrations: {
		path: 'prisma/migrations',
	},
	datasource: {
		// `?schema=frontend` is load-bearing. Prisma issues unqualified DDL and
		// lets the connection's search_path decide where it lands; without this
		// the tables follow the server default, which resolves `"$user"` to the
		// `gsf` schema because the role is also called `gsf`.
		url: `postgresql://${user}:${password}@${host}:${port}/${database}?schema=frontend`,
	},
});
