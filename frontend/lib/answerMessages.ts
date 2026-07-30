// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// One agent answer renders as up to two assistant bubbles (illumex-style):
// Message 1 — prose + SQL; Message 2 — charts, or the raw result table when
// visualization is off or produced nothing.
//
// The split lives here because two places have to agree on it exactly:
// `useChat.ts` renders it live, and the chat-completions proxy persists it in
// `after()`. If they disagreed, reloading a conversation would show a
// different layout than watching the answer arrive did.

import { stringifySqlResponse } from '@/lib/sqlResponse';

export const GENERIC_ANSWER_ERROR =
	'Something went wrong. Please try again, and if the issue persists, contact our support';

/** The agent keeps prose in `response` and chart specs in a separate array. */
export type AgentAnswer = {
	response?: string | null;
	sql_code?: string | null;
	sql_response_from_db?: unknown;
	charts?: unknown;
};

export type AnswerMessage = {
	content: string;
	sql?: string;
	sqlResponse?: string;
};

/** Strip ```chart / ```chart-carousel fences so Message 1 stays prose-only. */
export const stripChartFences = (markdown: string): string =>
	markdown
		.replace(/(^|\n)```(?:chart|chart-carousel)\b[\s\S]*?```/g, '\n')
		.replace(/\n{3,}/g, '\n\n')
		.trim();

export const chartsToFencedContent = (charts: Record<string, unknown>[]): string =>
	charts.map((spec) => `\`\`\`chart\n${JSON.stringify(spec)}\n\`\`\``).join('\n\n');

export const buildAnswerMessages = (answer: AgentAnswer): AnswerMessage[] => {
	const { response, sql_code: sqlCode, sql_response_from_db: sqlResponseFromDb, charts } = answer;
	const sqlResponse = stringifySqlResponse(sqlResponseFromDb);
	const prose = stripChartFences(response ?? '');
	const chartContent =
		Array.isArray(charts) && charts.length > 0
			? chartsToFencedContent(charts as Record<string, unknown>[])
			: null;

	const messages: AnswerMessage[] = [];

	if (prose || sqlCode) {
		messages.push({ content: prose, sql: sqlCode ?? undefined });
	}

	if (chartContent) {
		messages.push({ content: chartContent });
	} else if (sqlResponse) {
		messages.push({ content: '', sqlResponse });
	} else if (messages.length === 0) {
		// Nothing at all came back — surface something rather than silently
		// leaving the user without a reply.
		messages.push({ content: GENERIC_ANSWER_ERROR });
	}

	return messages;
};
