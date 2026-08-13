// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { NextResponse } from 'next/server';
import { getPrisma } from '@/lib/prisma';
import { withPermission } from '@/auth/with-auth';

type Ctx = { params: Promise<{ id: string }> };

// Fetch one custom prompt by id; 404 when no prompt has that id.
export const GET = withPermission<Ctx>({ prompt: ['read'] })(async (_req, { params }) => {
	const prisma = getPrisma();
	const { id } = await params;
	const prompt = await prisma.prompt.findUnique({ where: { id } });

	if (!prompt) {
		return NextResponse.json({ error: 'Custom prompt not found' }, { status: 404 });
	}

	return NextResponse.json(prompt);
});

// Update a custom prompt's text. `content` is the only writable field, and a
// body without it leaves the prompt unchanged.
export const PATCH = withPermission<Ctx>({ prompt: ['update'] })(async (req, { params }) => {
	const prisma = getPrisma();
	const { id } = await params;
	const body = await req.json();

	const data: Record<string, string> = {};
	if (typeof body.content === 'string') data.content = body.content;

	const prompt = await prisma.prompt.update({
		where: { id },
		data,
	});
	return NextResponse.json(prompt);
});

// Delete a custom prompt permanently. Answers 204 with no body.
export const DELETE = withPermission<Ctx>({ prompt: ['delete'] })(async (_req, { params }) => {
	const prisma = getPrisma();
	const { id } = await params;
	await prisma.prompt.delete({ where: { id } });
	return new Response(null, { status: 204 });
});
