// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export type ChatMessage = {
	id: string;
	role: 'user' | 'assistant';
	content: string;
	sql?: string;
	sqlResponse?: string;
	timestamp: number;
};

export type ChatRequest = {
	question: string;
	conversationId?: string | null;
	/** Scope retrieval/SQL to one connected database when multiple are loaded. */
	target_db?: string | null;
};

export type StepEvent = {
	type: 'step';
	node: string;
	label: string;
};

/** Shape of the executed SQL result, as returned by `sql_response_from_db`. */
export type SqlResult = string[] | { [key: string]: string }[];

export type ResultEvent = {
	type: 'result';
	answer: {
		response: string;
		sql_code?: string;
		sql_response_from_db?: SqlResult;
	};
};

export type ErrorEvent = {
	type: 'error';
	message: string;
};

export type ChatStreamEvent = StepEvent | ResultEvent | ErrorEvent;

/**
 * Illumex-style step 2: POST /api/chat/visualize takes the question, SQL, and
 * already-executed result from step 1 and returns ResultChart specs, or an
 * empty/null list when visualization is disabled or was skipped.
 *
 * Snake-cased because it is the FastAPI payload, which the Next proxy forwards
 * as-is; `conversation_id` is read by the proxy alone, to persist the bubble
 * this step produces and to keep two clients on one run from computing (and
 * writing) it twice.
 */
export type VisualizeRequest = {
	question: string;
	sql: string;
	result?: SqlResult;
	conversation_id?: string;
};

export type VisualizeResponse = {
	charts: Record<string, unknown>[] | null;
};

export type GraphStep = {
	node: string;
	label: string;
	status: 'completed' | 'active';
};

export type Conversation = {
	id: string;
	title: string;
	messages: ChatMessage[];
	createdAt: number;
};
