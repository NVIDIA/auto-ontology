// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ResolvedUser } from '@/auth/resolve-user';
import { getPrisma } from '@/lib/prisma';

// Shared by the watch, cancel, and visualization proxies. Each must confirm
// that a conversation belongs to the caller, and none should create one.
export const findOwnedConversation = async (
	user: ResolvedUser,
	conversationId: string,
): Promise<{ id: string } | null> => {
	const prisma = getPrisma();
	return prisma.conversation.findFirst({
		where: { id: conversationId, user_id: user.id },
		select: { id: true },
	});
};

// Conversation creation for completions is owned by FastAPI.
