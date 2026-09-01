// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { NextResponse } from 'next/server';
import { getPrisma } from '@/lib/prisma';
import { withPermission } from '@/auth/with-auth';
import { messageSelect } from '@/lib/apiSelects';

type Ctx = { params: Promise<{ id: string }> };

// Append a message to a conversation the caller owns.
//
// The manual path, for clients that record their own turns; the chat routes
// persist their messages themselves via FastAPI, so a normal chat flow never
// calls this. Also bumps the conversation's `updated_at`. 404 when the
// conversation does not exist or belongs to someone else.
export const POST = withPermission<Ctx>({ conversation: ['write'] })(async (
	req,
	{ params, user },
) => {
	const prisma = getPrisma();
	const { id } = await params;
	const body = await req.json();

	// Only allow adding messages to a conversation the user owns.
	const owned = await prisma.conversation.findFirst({
		where: { id, user_id: user.id },
		select: { id: true },
	});
	if (!owned) {
		return NextResponse.json({ error: 'Conversation not found' }, { status: 404 });
	}

	const message = await prisma.message.create({
		data: {
			conversation_id: id,
			role: body.role,
			content: body.content ?? '',
			sql_code: body.sql_code ?? null,
			sql_response: body.sql_response ?? null,
		},
		select: messageSelect,
	});

	await prisma.conversation.update({
		where: { id },
		data: { updated_at: new Date() },
	});

	return NextResponse.json(message, { status: 201 });
});
