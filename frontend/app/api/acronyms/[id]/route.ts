// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { NextResponse } from 'next/server';
import { getPrisma } from '@/lib/prisma';
import { requireApiAuth } from '@/auth/api-auth';

export async function PATCH(req: Request, { params }: { params: Promise<{ id: string }> }) {
	const denied = await requireApiAuth();
	if (denied) return denied;

	const prisma = getPrisma();
	const { id } = await params;
	const body = await req.json();

	const data: Record<string, string> = {};
	if (typeof body.name === 'string') data.name = body.name;
	if (typeof body.description === 'string') data.description = body.description;

	const acronym = await prisma.acronym.update({
		where: { id },
		data,
	});
	return NextResponse.json(acronym);
}

export async function DELETE(_req: Request, { params }: { params: Promise<{ id: string }> }) {
	const denied = await requireApiAuth();
	if (denied) return denied;

	const prisma = getPrisma();
	const { id } = await params;
	await prisma.acronym.delete({ where: { id } });
	return new Response(null, { status: 204 });
}
