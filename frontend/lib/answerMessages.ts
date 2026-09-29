// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// One agent answer renders as up to two assistant bubbles (illumex-style):
// Message 1 — prose + SQL; Message 2 — charts, or the raw result table when
// visualization is off or produced nothing.
//
// The two bubbles land at different times because charts are a separate step
// run after the SQL pipeline resolves: Message 1 ships as soon as the answer
// is ready, Message 2 once the chart step (run server-side, right inside
// `_pump` — see auto_ontology/server/chat/router.py) finishes and streams back as its
// own `charts` SSE event. FastAPI builds and persists both (Message 2's
// formatting lives in auto_ontology/server/chat/helpers.py now — there is no client-side
// equivalent to keep in sync anymore), so history always matches the live view
// regardless of whether any browser tab stuck around to watch.

import { stringifySqlResponse } from '@/lib/sqlResponse';

export const GENERIC_ANSWER_ERROR =
	'Something went wrong. Please try again, and if the issue persists, contact our support';

/** The agent's answer: prose in `response`, plus the SQL it ran and its result. */
export type AgentAnswer = {
	response?: string | null;
	sql_code?: string | null;
	sql_response_from_db?: unknown;
	thoughts?: string | null;
};

export type AnswerMessage = {
	content: string;
	sql?: string;
	sqlResponse?: string;
	thoughts?: string;
};

const hasChartFence = (content: string): boolean =>
	/(^|\n)```(?:chart|chart-carousel)\b/.test(content);

/** Strip ```chart / ```chart-carousel fences so Message 1 stays prose-only. */
export const stripChartFences = (markdown: string): string =>
	markdown
		.replace(/(^|\n)```(?:chart|chart-carousel)\b[\s\S]*?```/g, '\n')
		.replace(/\n{3,}/g, '\n\n')
		.trim();

/**
 * Whether `message` is a Message 2 — the bubble carrying the charts or the
 * result table. Recognising it is what tells "this turn is fully written" from
 * "Message 2 is still on its way" when polling history for a finished run.
 */
export const isResultMessage = (message: {
	role: string;
	content: string;
	sqlResponse?: string | null;
}): boolean =>
	message.role === 'assistant' && (message.sqlResponse != null || hasChartFence(message.content));

/** Message 1 — prose + SQL, ready the moment the agent's answer lands. */
export const buildSqlAnswerMessage = (answer: AgentAnswer): AnswerMessage | null => {
	const { response, sql_code: sqlCode, thoughts } = answer;
	const prose = stripChartFences(response ?? '');

	if (prose || sqlCode) {
		return { content: prose, sql: sqlCode ?? undefined, thoughts: thoughts ?? undefined };
	}

	// Nothing to say and no executed result for Message 2 to fall back on —
	// surface something rather than leaving the user without a reply.
	if (stringifySqlResponse(answer.sql_response_from_db) === undefined) {
		return { content: GENERIC_ANSWER_ERROR };
	}
	return null;
};
