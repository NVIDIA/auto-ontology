// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { NextResponse } from 'next/server';
import { getPrisma } from '@/lib/prisma';
import { withPermission } from '@/auth/with-auth';

export const GET = withPermission({ conversation: ['read'] })(async (_req, { user }) => {
	const prisma = getPrisma();
	const conversations = await prisma.conversation.findMany({
		where: { userId: user.id },
		orderBy: { createdAt: 'desc' },
	});
	return NextResponse.json(conversations);
});

export const POST = withPermission({ conversation: ['write'] })(async (req, { user }) => {
	const prisma = getPrisma();
	const body = await req.json();
	const conversation = await prisma.conversation.create({
		data: { title: body.title ?? '', userId: user.id },
	});
	return NextResponse.json(conversation, { status: 201 });
});
