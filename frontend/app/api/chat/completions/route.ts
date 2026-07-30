// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Route Handler that proxies POST /api/chat/completions to the FastAPI
// backend and streams the SSE response straight back to the browser.
//
// We use a hand-rolled handler (instead of Next.js `rewrites`) because the
// dev-server rewrites buffer streaming responses — the browser would receive
// nothing until the upstream connection closed, defeating SSE.
//
// This route is also the single writer of `Message` rows for chat turns —
// the user's turn is persisted synchronously below, the assistant's inside
// `after()`. Persisting here rather than letting the browser call
// `/api/conversations/[id]/messages` means a full turn is saved even if the
// user navigates away mid-stream — the agent keeps running server-side
// regardless (the analytics tee below keeps the upstream connection alive),
// so persistence should not depend on the browser still being around to see
// it finish.

import { after } from 'next/server';
import { withPermission } from '@/auth/with-auth';
import { userCan } from '@/auth/permissions';
import type { ResolvedUser } from '@/auth/resolve-user';
import { getPrisma } from '@/lib/prisma';
import { findOwnedConversation } from '@/lib/chatConversations';
import { buildAnswerMessages, type AgentAnswer, type AnswerMessage } from '@/lib/answerMessages';
import { isVisualizationEnabled } from '@/lib/configurations';

const PYTHON_API_URL = process.env.PYTHON_API_URL ?? 'http://127.0.0.1:3001';

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';
const SOURCE_HEADER = 'x-gsf-source';

const parseBody = (rawBody: string): Record<string, unknown> => {
	try {
		const parsed: unknown = JSON.parse(rawBody);
		return typeof parsed === 'object' && parsed !== null
			? (parsed as Record<string, unknown>)
			: {};
	} catch {
		return {};
	}
};

// Resolves `conversationId` to a conversation this user actually owns, gated
// on the same `conversation: ['write']` permission the messages endpoint
// enforces. Returns null (never throws) for any reason persistence should be
// skipped — missing id, not found/not owned, or lacking permission — so a
// bad/absent conversation_id degrades to "don't persist" rather than failing
// the chat request itself (this also preserves today's behavior for direct
// API/NAT-plugin callers that never send a conversation_id).
const resolveOwnedConversation = async (
	user: ResolvedUser,
	conversationId: string | null,
): Promise<{ id: string } | null> => {
	if (!conversationId) return null;
	if (!userCan(user, { conversation: ['write'] })) return null;
	return findOwnedConversation(user, conversationId);
};

type ChatStreamResultEvent = {
	type?: string;
	answer?: AgentAnswer;
	message?: string;
};

// Consume a teed copy of the SSE stream, returning the final answer (as the
// assistant bubbles it renders into), the analytics fields, and — on a
// mid-stream agent failure — the raw error message. Mirrors the parsing the
// browser does in `frontend/api/chat.ts` (data: lines, `[DONE]` sentinel,
// JSON `result`/`error` events).
const readFinalAnswer = async (
	stream: ReadableStream<Uint8Array>,
): Promise<{
	response: string | null;
	sql: string | null;
	answerMessages: AnswerMessage[];
	errorMessage: string | null;
}> => {
	const reader = stream.getReader();
	const decoder = new TextDecoder();
	let buffer = '';
	let response: string | null = null;
	let sql: string | null = null;
	let answerMessages: AnswerMessage[] = [];
	let errorMessage: string | null = null;

	const handleData = (data: string): void => {
		if (!data || data === '[DONE]') return;
		try {
			const event = JSON.parse(data) as ChatStreamResultEvent;
			if (event.type === 'result') {
				response = event.answer?.response ?? null;
				sql = event.answer?.sql_code ?? null;
				answerMessages = event.answer ? buildAnswerMessages(event.answer) : [];
			} else if (event.type === 'error') {
				errorMessage = event.message ?? null;
			}
		} catch {
			// Skip heartbeats / malformed lines.
		}
	};

	try {
		for (;;) {
			const { done, value } = await reader.read();
			if (done) break;
			buffer += decoder.decode(value, { stream: true });
			const lines = buffer.split('\n');
			buffer = lines.pop() ?? '';
			for (const line of lines) {
				const trimmed = line.trim();
				if (trimmed.startsWith('data:')) handleData(trimmed.slice(5).trim());
			}
		}
	} finally {
		reader.releaseLock();
	}

	return { response, sql, answerMessages, errorMessage };
};

export const POST = withPermission({ chat: ['use'] })(async (req, { user }) => {
	const payload = parseBody(await req.text());

	// The visualization toggle is an instance-wide setting, so it is resolved
	// here rather than trusted from the caller — a stale browser tab or a
	// direct API client cannot opt back into charts once admins turn them off.
	const body = JSON.stringify({
		...payload,
		visualization: await isVisualizationEnabled(),
	});

	const upstream = await fetch(`${PYTHON_API_URL}/api/chat/completions`, {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json',
			Accept: 'text/event-stream',
		},
		body,
		// Disable Node's transparent decompression so we can pipe bytes 1:1.
		// @ts-expect-error — `duplex` is required by Node's fetch when
		// streaming bodies; not yet in DOM lib types.
		duplex: 'half',
	});

	if (!upstream.ok || !upstream.body) {
		// Forward the upstream body verbatim so structured errors (e.g. the
		// 409 "Conversation in progress" payload from FastAPI) reach the
		// client and can be surfaced to the user. Fall back to a generic
		// message if the body could not be read.
		const contentType = upstream.headers.get('content-type') ?? 'text/plain';
		let errorBody: string;
		try {
			errorBody = await upstream.text();
		} catch {
			errorBody = `Upstream responded with ${upstream.status}`;
		}
		return new Response(errorBody || `Upstream responded with ${upstream.status}`, {
			status: upstream.status || 502,
			headers: { 'Content-Type': contentType },
		});
	}

	// Source label for the analytics row: the web app tags itself `app`; any
	// other caller (the NAT plugin / direct API) defaults to `api`.
	const source = req.headers.get(SOURCE_HEADER) ?? 'api';

	const responseHeaders = {
		'Content-Type': 'text/event-stream; charset=utf-8',
		'Cache-Control': 'no-cache, no-transform',
		Connection: 'keep-alive',
		'X-Accel-Buffering': 'no',
	};

	// Single writer for analytics and conversation history, for every caller.
	// Create the analytics row and persist the user's turn up front, then tee
	// the stream — one branch flows to the client untouched, the other is
	// parsed after the response to backfill the final answer and persist the
	// assistant's turn. `user` is the resolved GSF user (session or SSO
	// bearer), injected by withPermission.
	const question = typeof payload.question === 'string' ? payload.question : '';
	const conversationId =
		typeof payload.conversation_id === 'string' && payload.conversation_id.trim()
			? payload.conversation_id
			: null;

	const prisma = getPrisma();
	const [row, conversation] = await Promise.all([
		prisma.conversationAnalytics.create({
			data: { question, source, userId: user.id },
		}),
		resolveOwnedConversation(user, conversationId),
	]);

	if (conversation) {
		await prisma.message.create({
			data: { conversationId: conversation.id, role: 'user', content: question },
		});
	}

	const [toClient, toCapture] = upstream.body.tee();

	after(async () => {
		try {
			const { response, sql, answerMessages, errorMessage } =
				await readFinalAnswer(toCapture);
			await prisma.conversationAnalytics.update({
				where: { id: row.id },
				data: { response, sql, responseTimestamp: new Date() },
			});

			if (conversation) {
				// A result persists the same bubbles the browser rendered (prose +
				// SQL, then charts or the result table) so reopening the
				// conversation looks identical to watching it stream. A mid-stream
				// agent failure persists the real error text instead, paired with
				// the already-persisted user turn. A stream that closed with
				// neither (e.g. the upstream connection dropped) has nothing
				// meaningful to save.
				const rows: AnswerMessage[] =
					answerMessages.length > 0
						? answerMessages
						: errorMessage != null
							? [{ content: errorMessage }]
							: [];

				if (rows.length > 0) {
					await prisma.message.createMany({
						data: rows.map((msg) => ({
							conversationId: conversation.id,
							role: 'assistant',
							content: msg.content,
							sqlCode: msg.sql ?? null,
							sqlResponse: msg.sqlResponse ?? null,
						})),
					});
					await prisma.conversation.update({
						where: { id: conversation.id },
						data: { updatedAt: new Date() },
					});
				}
			}
		} catch {
			// Best-effort analytics/persistence — never let capture failures
			// affect the already-delivered chat response.
		}
	});

	return new Response(toClient, { status: 200, headers: responseHeaders });
});
