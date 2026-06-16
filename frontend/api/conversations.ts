// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Conversation, ChatMessage } from '@/types/chat';

export type ConversationSummary = {
	id: string;
	title: string;
	createdAt: string;
	updatedAt: string;
};

export type ConversationDetail = ConversationSummary & {
	messages: Array<{
		id: string;
		conversationId: string;
		role: 'user' | 'assistant';
		content: string;
		sqlCode: string | null;
		sqlResponse: string | null;
		createdAt: string;
	}>;
};

async function json<T>(input: RequestInfo, init?: RequestInit): Promise<T> {
	const res = await fetch(input, init);
	if (!res.ok) throw new Error(`API ${res.status}: ${res.statusText}`);
	return res.json() as Promise<T>;
}

export const conversationsApi = {
	list: () => json<ConversationSummary[]>('/api/conversations'),

	create: (title: string) =>
		json<ConversationSummary>('/api/conversations', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ title }),
		}),

	get: (id: string) => json<ConversationDetail>(`/api/conversations/${id}`),

	rename: (id: string, title: string) =>
		json<ConversationSummary>(`/api/conversations/${id}`, {
			method: 'PATCH',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ title }),
		}),

	delete: (id: string) => fetch(`/api/conversations/${id}`, { method: 'DELETE' }),

	addMessage: (
		conversationId: string,
		msg: {
			role: string;
			content: string;
			sqlCode?: string | null;
			sqlResponse?: string | null;
		},
	) =>
		json<{ id: string }>(`/api/conversations/${conversationId}/messages`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(msg),
		}),
};

export function toConversation(detail: ConversationDetail): Conversation {
	return {
		id: detail.id,
		title: detail.title,
		createdAt: new Date(detail.createdAt).getTime(),
		messages: detail.messages.map(toMessage),
	};
}

export function toMessage(msg: ConversationDetail['messages'][number]): ChatMessage {
	return {
		id: msg.id,
		role: msg.role,
		content: msg.content,
		sql: msg.sqlCode ?? undefined,
		sqlResponse: msg.sqlResponse ?? undefined,
		timestamp: new Date(msg.createdAt).getTime(),
	};
}
