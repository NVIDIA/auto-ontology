// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { NextResponse } from 'next/server';
import { getPrisma } from '@/lib/prisma';
import { withPermission } from '@/auth/with-auth';

type Ctx = { params: Promise<{ id: string }> };

export const POST = withPermission<Ctx>({ conversation: ['write'] })(async (
	req,
	{ params, user },
) => {
	const prisma = getPrisma();
	const { id } = await params;
	const body = await req.json();

	// Only allow adding messages to a conversation the user owns.
	const owned = await prisma.conversation.findFirst({ where: { id, userId: user.id } });
	if (!owned) {
		return NextResponse.json({ error: 'Conversation not found' }, { status: 404 });
	}

	const message = await prisma.message.create({
		data: {
			conversationId: id,
			role: body.role,
			content: body.content ?? '',
			sqlCode: body.sqlCode ?? null,
			sqlResponse: body.sqlResponse ?? null,
		},
	});

	await prisma.conversation.update({
		where: { id },
		data: { updatedAt: new Date() },
	});

	return NextResponse.json(message, { status: 201 });
});
