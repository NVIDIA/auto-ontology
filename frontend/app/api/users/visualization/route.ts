// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { NextResponse } from 'next/server';
import { getPrisma } from '@/lib/prisma';
import { withPermission } from '@/auth/with-auth';

// Per-user "Visualize SQL Results" preference (Settings > Agent Settings).
// Every user reads/updates only their own row — there is no cross-user access.
export const GET = withPermission({ visualization: ['read'] })(async (_req, { user }) => {
	const prisma = getPrisma();
	const row = await prisma.user.findUnique({
		where: { id: user.id },
		select: { visualization: true },
	});
	return NextResponse.json({ visualization: row?.visualization ?? true });
});

export const PUT = withPermission({ visualization: ['update'] })(async (req, { user }) => {
	const prisma = getPrisma();
	const body = await req.json();
	const visualization = body.visualization === true;

	await prisma.user.update({
		where: { id: user.id },
		data: { visualization },
	});

	return NextResponse.json({ visualization });
});
