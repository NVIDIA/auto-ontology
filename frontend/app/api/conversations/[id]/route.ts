// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { NextResponse } from 'next/server';
import { getPrisma } from '@/lib/prisma';
import { withPermission } from '@/auth/with-auth';
import { conversationSelect, messageSelect } from '@/lib/apiSelects';

type Ctx = { params: Promise<{ id: string }> };

const notFound = () => NextResponse.json({ error: 'Conversation not found' }, { status: 404 });

// Fetch one conversation with its full message history, oldest message first.
//
// Scoped to the caller, so another user's conversation reads as 404 rather than
// 403 — ownership is not disclosed.
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

// Rename a conversation the caller owns; 404 when it is not theirs.
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

// Delete a conversation and its messages; 404 when it is not the caller's.
//
// Answers 204 with no body. The delete cascades to the conversation's messages.
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
