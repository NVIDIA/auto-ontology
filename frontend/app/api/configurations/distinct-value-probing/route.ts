// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { NextResponse } from 'next/server';
import { getPrisma } from '@/lib/prisma';
import { DISTINCT_VALUE_PROBING_ENABLED_KEY, readOptOutFlag } from '@/lib/configurations';
import { withPermission } from '@/auth/with-auth';
import { distinctValueProbingBody } from './openapi';

// Key/value stored in the `configurations` table. The Python ingestion service
// reads this same key (see gsf/infra/feature_flags.py) once per compilation run
// to decide whether to run the per-column SELECT DISTINCT probes. It does NOT
// gate the bounded row sample, which always runs.
//
// Unlike semantic compilation, this flag is opt-OUT: an instance that has never
// touched the toggle still probes, which is what every deployment predating the
// toggle did. `readOptOutFlag` is what keeps this end in step with the Python
// reader, down to the whitespace and casing it tolerates.
const CONFIG_KEY = DISTINCT_VALUE_PROBING_ENABLED_KEY;

// Report whether compilation scans low-cardinality columns for distinct values.
export const GET = withPermission({ semanticCompilation: ['read'] })(async () => {
	const prisma = getPrisma();
	const row = await prisma.configuration.findUnique({ where: { key: CONFIG_KEY } });
	return NextResponse.json({ enabled: readOptOutFlag(row?.value) });
});

// Turn distinct value scanning on or off instance-wide (admin-only).
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
// default, so `{"enabled": "true"}` from a shell client would disable scanning
// while reading as if it enabled it.
export const PUT = withPermission({ semanticCompilation: ['manage'] })(async (req) => {
	const prisma = getPrisma();

	let raw: unknown;
	try {
		raw = await req.json();
	} catch {
		return NextResponse.json({ error: 'Body must be JSON.' }, { status: 400 });
	}

	const parsed = distinctValueProbingBody.safeParse(raw);
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
