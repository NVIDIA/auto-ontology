// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { NextResponse } from 'next/server';
import { getPrisma } from '@/lib/prisma';
import { getApiUser } from '@/auth/api-auth';

const notFound = () => NextResponse.json({ error: 'Conversation not found' }, { status: 404 });

export async function GET(_req: Request, { params }: { params: Promise<{ id: string }> }) {
	const { userId, deny } = await getApiUser();
	if (deny) return deny;

	const prisma = getPrisma();
	const { id } = await params;
	// Scope by userId so one user can't read another's conversation.
	const conversation = await prisma.conversation.findFirst({
		where: { id, userId },
		include: { messages: { orderBy: { createdAt: 'asc' } } },
	});

	if (!conversation) {
		return notFound();
	}

	return NextResponse.json(conversation);
}

export async function PATCH(req: Request, { params }: { params: Promise<{ id: string }> }) {
	const { userId, deny } = await getApiUser();
	if (deny) return deny;

	const prisma = getPrisma();
	const { id } = await params;
	const body = await req.json();

	// updateMany so the userId filter applies; count tells us if it was owned.
	const result = await prisma.conversation.updateMany({
		where: { id, userId },
		data: { title: body.title },
	});
	if (result.count === 0) {
		return notFound();
	}

	const conversation = await prisma.conversation.findUnique({ where: { id } });
	return NextResponse.json(conversation);
}

export async function DELETE(_req: Request, { params }: { params: Promise<{ id: string }> }) {
	const { userId, deny } = await getApiUser();
	if (deny) return deny;

	const prisma = getPrisma();
	const { id } = await params;
	const result = await prisma.conversation.deleteMany({ where: { id, userId } });
	if (result.count === 0) {
		return notFound();
	}
	return new Response(null, { status: 204 });
}
