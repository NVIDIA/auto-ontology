// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { NextResponse } from 'next/server';
import { getPrisma } from '@/lib/prisma';

// Analytics always cover a fixed trailing window; not configurable per-request.
const ANALYTICS_DAYS = 30;

const parseIntParam = (value: string | null, fallback: number): number => {
	if (value == null) return fallback;
	const parsed = Number.parseInt(value, 10);
	return Number.isFinite(parsed) ? parsed : fallback;
};

export async function GET(request: Request) {
	const prisma = getPrisma();
	const { searchParams } = new URL(request.url);

	const skip = parseIntParam(searchParams.get('skip'), 0);
	const limitParam = searchParams.get('limit');
	const limit = limitParam != null ? parseIntParam(limitParam, 0) : null;

	const cutoff = new Date(Date.now() - ANALYTICS_DAYS * 24 * 60 * 60 * 1000);
	const where = { questionTimestamp: { gte: cutoff } };

	const total = await prisma.conversationAnalytics.count({ where });
	const data = await prisma.conversationAnalytics.findMany({
		where,
		orderBy: { questionTimestamp: 'desc' },
		skip,
		...(limit != null ? { take: limit } : {}),
	});

	return NextResponse.json({ data, total });
}

export async function POST(req: Request) {
	const prisma = getPrisma();
	const body = await req.json();
	const row = await prisma.conversationAnalytics.create({
		data: {
			questionMessageId: body.questionMessageId,
			question: body.question ?? '',
		},
	});
	return NextResponse.json(row, { status: 201 });
}
