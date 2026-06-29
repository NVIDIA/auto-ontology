// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { NextResponse } from 'next/server';
import { getPrisma } from '@/lib/prisma';
import { getApiUser } from '@/auth/api-auth';

export async function GET() {
	const { userId, deny } = await getApiUser();
	if (deny) return deny;

	const prisma = getPrisma();
	const conversations = await prisma.conversation.findMany({
		where: { userId },
		orderBy: { createdAt: 'desc' },
	});
	return NextResponse.json(conversations);
}

export async function POST(req: Request) {
	const { userId, deny } = await getApiUser();
	if (deny) return deny;

	const prisma = getPrisma();
	const body = await req.json();
	const conversation = await prisma.conversation.create({
		data: { title: body.title ?? '', userId },
	});
	return NextResponse.json(conversation, { status: 201 });
}
