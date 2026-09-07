// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { NextResponse } from 'next/server';
import { getPrisma } from '@/lib/prisma';
import { withPermission } from '@/auth/with-auth';

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
export const PUT = withPermission({ semanticCompilation: ['manage'] })(async (req) => {
	const prisma = getPrisma();
	const body = await req.json();
	const enabled = body.enabled === true;
	const value = enabled ? 'true' : 'false';

	await prisma.configuration.upsert({
		where: { key: CONFIG_KEY },
		create: { key: CONFIG_KEY, value },
		update: { value },
	});

	return NextResponse.json({ enabled });
});
