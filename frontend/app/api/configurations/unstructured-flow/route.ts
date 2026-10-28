// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { NextResponse } from 'next/server';
import { z } from 'zod';
import { withPermission } from '@/auth/with-auth';
import { UNSTRUCTURED_FLOW_ENABLED_KEY, isUnstructuredFlowEnabled } from '@/lib/configurations';
import { getPrisma } from '@/lib/prisma';

const unstructuredFlowBody = z.object({ enabled: z.boolean() });

export const GET = withPermission({ document: ['read'] })(async () => {
	return NextResponse.json({ enabled: await isUnstructuredFlowEnabled() });
});

export const PUT = withPermission({ document: ['manage'] })(async (req) => {
	let raw: unknown;
	try {
		raw = await req.json();
	} catch {
		return NextResponse.json({ error: 'Body must be JSON.' }, { status: 400 });
	}

	const parsed = unstructuredFlowBody.safeParse(raw);
	if (!parsed.success) {
		return NextResponse.json({ error: '`enabled` must be a boolean.' }, { status: 400 });
	}

	const { enabled } = parsed.data;
	const value = enabled ? 'true' : 'false';
	await getPrisma().configuration.upsert({
		where: { key: UNSTRUCTURED_FLOW_ENABLED_KEY },
		create: { key: UNSTRUCTURED_FLOW_ENABLED_KEY, value },
		update: { value },
	});

	return NextResponse.json({ enabled });
});
