// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { NextResponse } from 'next/server';
import { getPrisma } from '@/lib/prisma';
import { PII_DETECTION_ENABLED_KEY, isPiiDetectionEnabled } from '@/lib/configurations';
import { withPermission } from '@/auth/with-auth';
import { piiDetectionBody } from './openapi';

// Key/value stored in the `configurations` table. The Python ingestion service
// reads this same key (see auto_ontology/infra/feature_flags.py) once per ingest
// to decide whether to classify catalog columns and attach the shared PII tag.
//
// The flag is opt-IN: an instance that has never touched the toggle does not
// classify. `isPiiDetectionEnabled` is what keeps this end in step with the
// Python reader, down to the whitespace and casing it tolerates.
const CONFIG_KEY = PII_DETECTION_ENABLED_KEY;

// Report whether ingest classifies catalog columns as PII.
export const GET = withPermission({ pii: ['read'] })(async () => {
	return NextResponse.json({ enabled: await isPiiDetectionEnabled() });
});

// Turn PII detection on or off instance-wide (admin-only).
//
// Deliberately does NOT trigger an ingest. The flag is read at the start of
// each ingest's PII step, so it applies from the next one onward — flipping it
// is a change of policy, not a request to re-scan. Existing PII tags are left
// in place.
//
// Rejects a non-boolean `enabled` rather than coercing it, so
// `{"enabled": "true"}` from a shell client cannot silently disagree with
// what the page would show after a save.
export const PUT = withPermission({ pii: ['manage'] })(async (req) => {
	const prisma = getPrisma();

	let raw: unknown;
	try {
		raw = await req.json();
	} catch {
		return NextResponse.json({ error: 'Body must be JSON.' }, { status: 400 });
	}

	const parsed = piiDetectionBody.safeParse(raw);
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
