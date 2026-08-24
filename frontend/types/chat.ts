// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export type ChatMessage = {
	id: string;
	role: 'user' | 'assistant';
	content: string;
	sql?: string;
	sqlResponse?: string;
	thoughts?: string;
	timestamp: number;
};

export type ChatRequest = {
	question: string;
	/**
	 * Stable thread UUID. When supplied, prior completed turns are loaded and
	 * the new turn — including the chart/table bubble, once ready — is
	 * persisted to it. An unknown UUID creates the conversation. Omitting it
	 * runs the question statelessly, with no chart step or history.
	 */
	conversationId?: string | null;
	/** Scope retrieval/SQL to one connected database when multiple are loaded. */
	target_db?: string | null;
};

export type StepEvent = {
	type: 'step';
	node: string;
	label: string;
	thought?: string | null;
};

/** Shape of the executed SQL result, as returned by `sql_response_from_db`. */
export type SqlResult = string[] | { [key: string]: string }[];

export type ResultEvent = {
	type: 'result';
	answer: {
		response: string;
		sql_code?: string;
		sql_response_from_db?: SqlResult;
		thoughts?: string;
	};
};

export type ErrorEvent = {
	type: 'error';
	message: string;
};

/**
 * Message 2 — the chart or fallback result table, generated and persisted
 * server-side (see `gsf/server/chat/router.py`'s `_pump`) right after the SQL
 * answer, using the same guarantee as Message 1: it exists whether or not a
 * browser tab is still around to ask for it. Omitted entirely when there was
 * no executed result to visualize.
 */
export type ChartsEvent = {
	type: 'charts';
	content: string;
	sql_response?: string | null;
};

export type ChatStreamEvent = StepEvent | ResultEvent | ErrorEvent | ChartsEvent;

export type GraphStep = {
	node: string;
	label: string;
	thought?: string | null;
	status: 'completed' | 'active';
};

export type Conversation = {
	id: string;
	title: string;
	messages: ChatMessage[];
	createdAt: number;
};
