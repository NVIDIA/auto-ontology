// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { NextResponse } from 'next/server';
import { getPrisma } from '@/lib/prisma';
import { requireApiAuth } from '@/auth/api-auth';

export async function GET(req: Request) {
	const denied = await requireApiAuth();
	if (denied) return denied;

	const prisma = getPrisma();
	const { searchParams } = new URL(req.url);
	const name = searchParams.get('name')?.trim();

	if (name) {
		const existing = await prisma.acronym.findUnique({ where: { name } });
		return NextResponse.json({ exists: existing !== null });
	}

	const acronyms = await prisma.acronym.findMany({ orderBy: { createdAt: 'desc' } });
	return NextResponse.json(acronyms);
}

export async function POST(req: Request) {
	const denied = await requireApiAuth();
	if (denied) return denied;

	const prisma = getPrisma();
	const body = await req.json();
	const acronym = await prisma.acronym.create({
		data: {
			name: body.name,
			description: body.description ?? '',
		},
	});
	return NextResponse.json(acronym, { status: 201 });
}
