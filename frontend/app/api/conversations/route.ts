// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { NextResponse } from 'next/server';
import { getPrisma } from '@/lib/prisma';
import { withPermission } from '@/auth/with-auth';
import { conversationSelect } from '@/lib/apiSelects';

// List the caller's own conversations, newest first, without their messages.
//
// Scoped to the authenticated user: a conversation is only ever visible to its
// owner, so there is no way to list someone else's.
export const GET = withPermission({ conversation: ['read'] })(async (_req, { user }) => {
	const prisma = getPrisma();
	const conversations = await prisma.conversation.findMany({
		where: { user_id: user.id },
		orderBy: { created_at: 'desc' },
		select: conversationSelect,
	});
	return NextResponse.json(conversations);
});

// Start a new, empty conversation owned by the caller.
//
// `title` is optional and defaults to empty. Chat clients usually create the
// conversation here first, then pass its id to `POST /api/chat/completions`.
export const POST = withPermission({ conversation: ['write'] })(async (req, { user }) => {
	const prisma = getPrisma();
	const body = await req.json();
	const conversation = await prisma.conversation.create({
		data: { title: body.title ?? '', user_id: user.id },
		select: conversationSelect,
	});
	return NextResponse.json(conversation, { status: 201 });
});
