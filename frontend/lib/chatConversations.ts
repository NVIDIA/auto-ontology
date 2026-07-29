// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ResolvedUser } from '@/auth/resolve-user';
import { getPrisma } from '@/lib/prisma';

// Shared by the chat-completions proxy (persists turns into a conversation)
// and the chat-watch proxy (reattaches to a run already in progress) — both
// need to confirm a conversation_id actually belongs to the calling user
// before touching it.
export const findOwnedConversation = async (
	user: ResolvedUser,
	conversationId: string,
): Promise<{ id: string } | null> => {
	const prisma = getPrisma();
	return prisma.conversation.findFirst({
		where: { id: conversationId, userId: user.id },
		select: { id: true },
	});
};
