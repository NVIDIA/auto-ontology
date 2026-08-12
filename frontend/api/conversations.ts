// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Conversation, ChatMessage } from '@/types/chat';

export type ConversationSummary = {
	id: string;
	title: string;
	created_at: string;
	updated_at: string;
};

export type ConversationDetail = ConversationSummary & {
	messages: Array<{
		id: string;
		conversation_id: string;
		role: 'user' | 'assistant';
		content: string;
		sql_code: string | null;
		sql_response: string | null;
		created_at: string;
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
			// Callers pass the camelCase shape the chat UI works in; the API
			// speaks snake_case, so the rename happens here rather than at
			// every call site.
			body: JSON.stringify({
				role: msg.role,
				content: msg.content,
				sql_code: msg.sqlCode ?? null,
				sql_response: msg.sqlResponse ?? null,
			}),
		}),
};

export function toConversation(detail: ConversationDetail): Conversation {
	return {
		id: detail.id,
		title: detail.title,
		createdAt: new Date(detail.created_at).getTime(),
		messages: detail.messages.map(toMessage),
	};
}

export function toMessage(msg: ConversationDetail['messages'][number]): ChatMessage {
	return {
		id: msg.id,
		role: msg.role,
		content: msg.content,
		sql: msg.sql_code ?? undefined,
		sqlResponse: msg.sql_response ?? undefined,
		timestamp: new Date(msg.created_at).getTime(),
	};
}
