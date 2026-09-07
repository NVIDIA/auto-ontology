// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { NextResponse } from 'next/server';
import { getPrisma } from '@/lib/prisma';
import { withPermission } from '@/auth/with-auth';
import { columnProfilingBody } from './openapi';

// Key/value stored in the `configurations` table. The Python ingestion service
// reads this same key (see gsf/infra/feature_flags.py) once per compilation run
// to decide whether to sample live column values.
const CONFIG_KEY = 'column_profiling_enabled';

// Unlike semantic compilation, this flag is opt-OUT: an instance that has never
// touched the toggle profiles column values, which is what every deployment
// predating the toggle did. Absent row therefore reads as `true`, and the
// backend default in feature_flags.py has to agree.
const readEnabled = (value: string | undefined) => value !== 'false';

// Report whether semantic compilation samples live column values.
export const GET = withPermission({ semanticCompilation: ['read'] })(async () => {
	const prisma = getPrisma();
	const row = await prisma.configuration.findUnique({ where: { key: CONFIG_KEY } });
	return NextResponse.json({ enabled: readEnabled(row?.value) });
});

// Turn column profiling on or off instance-wide (admin-only).
//
// Deliberately does NOT trigger a compilation run. The flag is read at the top
// of each run, so it applies from the next one onward — flipping it is a change
// of policy, not a request to recompile. Operators who want it applied now can
// use Reset, which is on the same page.
//
// Rejects a non-boolean `enabled` rather than coercing it, which is why this
// differs from the sibling configuration routes. Those flags are opt-in, so
// coercing a malformed value to false lands on their default and is harmless.
// This one is opt-out: coercion would silently turn OFF something that is on by
// default, so `{"enabled": "true"}` from a shell client would disable profiling
// while reading as if it enabled it.
export const PUT = withPermission({ semanticCompilation: ['manage'] })(async (req) => {
	const prisma = getPrisma();

	let raw: unknown;
	try {
		raw = await req.json();
	} catch {
		return NextResponse.json({ error: 'Body must be JSON.' }, { status: 400 });
	}

	const parsed = columnProfilingBody.safeParse(raw);
	if (!parsed.success) {
		return NextResponse.json({ error: '`enabled` must be a boolean.' }, { status: 400 });
	}

	const { enabled } = parsed.data;
	const value = enabled ? 'true' : 'false';

	await prisma.configuration.upsert({
		where: { key: CONFIG_KEY },
		create: { key: CONFIG_KEY, value },
		update: { value },
	});

	return NextResponse.json({ enabled });
});
