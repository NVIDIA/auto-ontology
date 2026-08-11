// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { NextResponse } from 'next/server';
import { getPrisma } from '@/lib/prisma';
import { withPermission } from '@/auth/with-auth';
import { conversationSelect, messageSelect } from '@/lib/apiSelects';

type Ctx = { params: Promise<{ id: string }> };

const notFound = () => NextResponse.json({ error: 'Conversation not found' }, { status: 404 });

export const GET = withPermission<Ctx>({ conversation: ['read'] })(async (
	_req,
	{ params, user },
) => {
	const prisma = getPrisma();
	const { id } = await params;
	// Scope by userId so one user can't read another's conversation.
	const conversation = await prisma.conversation.findFirst({
		where: { id, user_id: user.id },
		select: {
			...conversationSelect,
			messages: { select: messageSelect, orderBy: { created_at: 'asc' } },
		},
	});

	if (!conversation) {
		return notFound();
	}

	return NextResponse.json(conversation);
});

export const PATCH = withPermission<Ctx>({ conversation: ['write'] })(async (
	req,
	{ params, user },
) => {
	const prisma = getPrisma();
	const { id } = await params;
	const body = await req.json();

	// updateMany so the userId filter applies; count tells us if it was owned.
	const result = await prisma.conversation.updateMany({
		where: { id, user_id: user.id },
		data: { title: body.title },
	});
	if (result.count === 0) {
		return notFound();
	}

	const conversation = await prisma.conversation.findUnique({
		where: { id },
		select: conversationSelect,
	});
	return NextResponse.json(conversation);
});

export const DELETE = withPermission<Ctx>({ conversation: ['delete'] })(async (
	_req,
	{ params, user },
) => {
	const prisma = getPrisma();
	const { id } = await params;
	const result = await prisma.conversation.deleteMany({ where: { id, user_id: user.id } });
	if (result.count === 0) {
		return notFound();
	}
	return new Response(null, { status: 204 });
});
