// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { NextResponse } from 'next/server';
import { getPrisma } from '@/lib/prisma';
import { withPermission } from '@/auth/with-auth';
import { Role } from '@/enums/auth';

// Analytics always cover a fixed trailing window; not configurable per-request.
const ANALYTICS_DAYS = 30;

const parseIntParam = (value: string | null, fallback: number): number => {
	if (value == null) return fallback;
	const parsed = Number.parseInt(value, 10);
	return Number.isFinite(parsed) ? parsed : fallback;
};

// Viewing the analytics report requires the analytics:read permission (admin).
export const GET = withPermission({ analytics: ['read'] })(async (request) => {
	const prisma = getPrisma();
	const { searchParams } = new URL(request.url);

	const skip = parseIntParam(searchParams.get('skip'), 0);
	const limitParam = searchParams.get('limit');
	const limit = limitParam != null ? parseIntParam(limitParam, 0) : null;

	const cutoff = new Date(Date.now() - ANALYTICS_DAYS * 24 * 60 * 60 * 1000);
	const where = { questionTimestamp: { gte: cutoff } };

	const total = await prisma.conversationAnalytics.count({ where });
	// The row stores only `userId`; join the User to resolve the display name.
	const rows = await prisma.conversationAnalytics.findMany({
		where,
		orderBy: { questionTimestamp: 'desc' },
		skip,
		...(limit != null ? { take: limit } : {}),
		include: { user: { select: { id: true, name: true, email: true, role: true } } },
	});

	const data = rows.map((row) => ({
		...row,
		user: { ...row.user, role: (row.user.role as Role) ?? Role.Viewer },
	}));

	return NextResponse.json({ data, total });
});
