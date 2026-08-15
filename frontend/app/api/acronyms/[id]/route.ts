// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { NextResponse } from 'next/server';
import { getPrisma } from '@/lib/prisma';
import { withPermission } from '@/auth/with-auth';
import { acronymSelect } from '@/lib/apiSelects';

type Ctx = { params: Promise<{ id: string }> };

// Update an acronym's name or description.
//
// Both fields are optional and only the ones present in the body are written,
// so a partial body leaves the rest of the record untouched.
export const PATCH = withPermission<Ctx>({ acronym: ['update'] })(async (req, { params }) => {
	const prisma = getPrisma();
	const { id } = await params;
	const body = await req.json();

	const data: Record<string, string> = {};
	if (typeof body.name === 'string') data.name = body.name;
	if (typeof body.description === 'string') data.description = body.description;

	const acronym = await prisma.acronym.update({
		where: { id },
		data,
		select: acronymSelect,
	});
	return NextResponse.json(acronym);
});

// Delete an acronym permanently. Answers 204 with no body.
export const DELETE = withPermission<Ctx>({ acronym: ['delete'] })(async (_req, { params }) => {
	const prisma = getPrisma();
	const { id } = await params;
	await prisma.acronym.delete({ where: { id } });
	return new Response(null, { status: 204 });
});
