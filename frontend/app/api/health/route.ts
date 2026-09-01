// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { NextResponse } from 'next/server';
import { getPrisma } from '@/lib/prisma';
import { withPublic } from '@/auth/with-auth';

export const dynamic = 'force-dynamic';

type CheckResult = { status: 'ok' } | { status: 'error'; detail: string };

async function checkPostgres(): Promise<CheckResult> {
	try {
		await getPrisma().$queryRaw`SELECT 1`;
		return { status: 'ok' };
	} catch (err) {
		const detail = err instanceof Error ? err.message : String(err);
		return { status: 'error', detail: detail.slice(0, 200) };
	}
}

// Liveness/readiness probe for the Next.js tier, unauthenticated by design.
//
// Probes Postgres and answers 200 `{ status: 'ok' }`, or 503
// `{ status: 'degraded' }` carrying the failing check's error detail.
export const GET = withPublic(async () => {
	const postgres = await checkPostgres();
	const healthy = postgres.status === 'ok';
	return NextResponse.json(
		{ status: healthy ? 'ok' : 'degraded', postgres },
		{ status: healthy ? 200 : 503 },
	);
});
