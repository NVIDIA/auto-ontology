// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { NextResponse } from 'next/server';
import { getPrisma } from '@/lib/prisma';
import { SQL_QUERY_TIMEOUT_SECONDS_KEY } from '@/lib/configurations';
import {
	MAX_SQL_QUERY_TIMEOUT_SECONDS,
	MIN_SQL_QUERY_TIMEOUT_SECONDS,
	readSqlQueryTimeout,
} from '@/lib/sqlQueryTimeout';
import { withPermission } from '@/auth/with-auth';
import { sqlQueryTimeoutBody } from './openapi';

// Key/value stored in the `configurations` table. The text-to-SQL pipeline
// reads this same key (see auto_ontology/infra/feature_flags.py) before each
// statement it issues, and cancels the statement once it runs this long. An
// absent row means 30 seconds, which is also what a backend running without
// this frontend uses.

// Report the instance-wide SQL query timeout, in seconds.
export const GET = withPermission({ visualization: ['read'] })(async () => {
	const row = await getPrisma().configuration.findUnique({
		where: { key: SQL_QUERY_TIMEOUT_SECONDS_KEY },
	});
	return NextResponse.json({ seconds: readSqlQueryTimeout(row?.value) });
});

// Set the instance-wide SQL query timeout (admin-only).
//
// Rejects anything but an in-range integer rather than coercing it: the backend
// treats an unusable stored value as the default, so a coerced value would be
// silently ignored while the page showed it as saved.
export const PUT = withPermission({ visualization: ['manage'] })(async (req) => {
	let raw: unknown;
	try {
		raw = await req.json();
	} catch {
		return NextResponse.json({ error: 'Body must be JSON.' }, { status: 400 });
	}

	const parsed = sqlQueryTimeoutBody.safeParse(raw);
	if (!parsed.success) {
		return NextResponse.json(
			{
				error: `Timeout must be a whole number of seconds from ${MIN_SQL_QUERY_TIMEOUT_SECONDS} to ${MAX_SQL_QUERY_TIMEOUT_SECONDS}.`,
			},
			{ status: 400 },
		);
	}

	const { seconds } = parsed.data;
	const value = String(seconds);

	await getPrisma().configuration.upsert({
		where: { key: SQL_QUERY_TIMEOUT_SECONDS_KEY },
		create: { key: SQL_QUERY_TIMEOUT_SECONDS_KEY, value },
		update: { value },
	});

	return NextResponse.json({ seconds });
});
