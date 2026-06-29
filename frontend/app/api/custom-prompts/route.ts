// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { NextResponse } from 'next/server';
import { getPrisma } from '@/lib/prisma';
import { requireApiAuth } from '@/auth/api-auth';

export async function GET() {
	const denied = await requireApiAuth();
	if (denied) return denied;

	const prisma = getPrisma();
	const prompts = await prisma.prompt.findMany();
	return NextResponse.json(prompts);
}

export async function POST(req: Request) {
	const denied = await requireApiAuth();
	if (denied) return denied;

	const prisma = getPrisma();
	const body = await req.json();
	const prompt = await prisma.prompt.create({
		data: {
			content: body.content ?? '',
		},
	});
	return NextResponse.json(prompt, { status: 201 });
}
