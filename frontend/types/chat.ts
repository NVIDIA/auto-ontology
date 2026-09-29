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

/**
 * A graph node starting (`start`) or returning (`end`). Both are sent for
 * every node: the backend can only attach a node's `thought` once it has
 * finished, but waiting for that to show the label would leave the UI naming
 * the *previous* node for the whole time this one runs — a 20s reconstruction
 * displayed as "Validating intent". `phase` is absent on older backends, in
 * which case an event is treated as a plain "next step" as before.
 */
export type StepEvent = {
	type: 'step';
	node: string;
	label: string;
	phase?: 'start' | 'end';
	thought?: string | null;
};

/**
 * The query the agent is about to run, sent once it has cleared syntax and
 * intent validation, so the UI can show it while the database works rather
 * than only alongside the final answer. Drafts that validation rejects are
 * never sent, so what arrives here is always what executes. A query that
 * fails at execution and is rebuilt sends a new event; the newest one wins.
 */
export type SqlEvent = {
	type: 'sql';
	node: string;
	sql: string;
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
 * server-side (see `auto_ontology/server/chat/router.py`'s `_pump`) right after the SQL
 * answer, using the same guarantee as Message 1: it exists whether or not a
 * browser tab is still around to ask for it. Omitted entirely when there was
 * no executed result to visualize.
 */
export type ChartsEvent = {
	type: 'charts';
	content: string;
	sql_response?: string | null;
};

export type ChatStreamEvent = StepEvent | SqlEvent | ResultEvent | ErrorEvent | ChartsEvent;

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
