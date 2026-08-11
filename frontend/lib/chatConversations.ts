// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { userCan } from '@/auth/permissions';
import type { ResolvedUser } from '@/auth/resolve-user';
import { getPrisma } from '@/lib/prisma';

// Shared by the chat-watch and chat-cancel proxies (reattaching to / aborting
// a run already in progress) — both need to confirm a conversation_id
// actually belongs to the calling user before touching it, and neither
// should ever create one: there's nothing to watch or cancel for a
// conversation that was never running.
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

// Used by the chat-completions proxy to persist turns into a conversation
// even when the caller (typically a direct API/NAT-plugin caller that never
// went through `POST /api/conversations`) sends a `conversation_id` that
// doesn't exist yet. Visualize deliberately does not use this — by step 2
// the conversation should already exist, and create-on-missing there would
// leave an orphan chart/table-only conversation. Missing id → null (nothing
// to create); not found → create it for this user; found but owned by
// someone else → null (never hijack it).
export const resolveOrCreateOwnedConversation = async (
	user: ResolvedUser,
	conversationId: string | null,
	title: string,
): Promise<{ id: string } | null> => {
	if (!conversationId) return null;
	if (!userCan(user, { conversation: ['write'] })) return null;

	try {
		const existing = await findOwnedConversation(user, conversationId);
		if (existing) return existing;

		const prisma = getPrisma();
		const clash = await prisma.conversation.findUnique({
			where: { id: conversationId },
			select: { id: true },
		});
		if (clash) return null;

		return await prisma.conversation.create({
			data: { id: conversationId, user_id: user.id, title },
			select: { id: true },
		});
	} catch {
		// Malformed id (not a UUID — the column is `@db.Uuid`) or a create/create
		// race with a concurrent request for the same new id: degrade to "don't
		// persist" rather than failing the chat response.
		return null;
	}
};
