// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { NextResponse } from 'next/server';
import { getPrisma } from '@/lib/prisma';
import { withPermission } from '@/auth/with-auth';

// Key/value stored in the `configurations` table. The Python ingestion service
// reads this same key to decide whether to start its semantic scheduler.
const CONFIG_KEY = 'semantic_compilation_enabled';

const PYTHON_API_URL = process.env.PYTHON_API_URL ?? 'http://127.0.0.1:3001';

// Report whether scheduled semantic compilation is enabled instance-wide.
export const GET = withPermission({ semanticCompilation: ['read'] })(async () => {
	const prisma = getPrisma();
	const row = await prisma.configuration.findUnique({ where: { key: CONFIG_KEY } });
	return NextResponse.json({ enabled: row?.value === 'true' });
});

// Turn scheduled semantic compilation on or off instance-wide (admin-only).
//
// Enabling also kicks off a compilation run immediately and starts the ingestion
// service's scheduler, so the change takes effect without a restart. That
// trigger is best-effort: the setting is still saved if the service is down, and
// the scheduler picks the flag up on its next boot.
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

	// On enable, kick off a compilation run now (the ingestion service also
	// starts its scheduler on this call, so it takes effect without a restart).
	// Best-effort: a save must still succeed if the ingestion service is down.
	if (enabled) {
		try {
			await fetch(`${PYTHON_API_URL}/api/semantic-compilation/trigger`, {
				method: 'POST',
				headers: { Accept: 'application/json' },
			});
		} catch {
			// Swallow — the flag is persisted; the scheduler picks it up on next boot.
		}
	}

	return NextResponse.json({ enabled });
});
