// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { NextResponse } from 'next/server';
import { getPrisma } from '@/lib/prisma';
import { withPermission } from '@/auth/with-auth';

// List every custom prompt: the instance-wide prompt overrides the agent uses.
export const GET = withPermission({ prompt: ['read'] })(async () => {
	const prisma = getPrisma();
	const prompts = await prisma.prompt.findMany();
	return NextResponse.json(prompts);
});

// Create a custom prompt from `content`, which defaults to empty when omitted.
export const POST = withPermission({ prompt: ['create'] })(async (req) => {
	const prisma = getPrisma();
	const body = await req.json();
	const prompt = await prisma.prompt.create({
		data: {
			content: body.content ?? '',
		},
	});
	return NextResponse.json(prompt, { status: 201 });
});
