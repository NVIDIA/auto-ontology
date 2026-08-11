// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Second step: proxies POST /api/chat/visualize to the FastAPI backend once
// the client already has the SQL + executed result from step 1
// (POST /api/chat/completions). Kept as a separate, non-streaming request so
// the answer never waits on an extra LLM round trip just to pick a chart.
//
// This route also persists the assistant bubble it produces (charts, or the
// result table when there are none), because that bubble only exists once the
// chart step resolves — the completions proxy has long since written Message 1
// and cannot know how this one turns out.
//
// Only the client that submitted the question calls this, so one turn produces
// exactly one request and one Message 2 — no dedupe needed here. A tab watching
// someone else's run (via /api/chat/watch) deliberately stops after Message 1
// and picks this bubble up from history on the next load.

import { NextResponse } from 'next/server';
import { withPermission } from '@/auth/with-auth';
import { getPrisma } from '@/lib/prisma';
import { findOwnedConversation } from '@/lib/chatConversations';
import { buildResultMessage } from '@/lib/answerMessages';
import { isVisualizationEnabled } from '@/lib/configurations';

const PYTHON_API_URL = process.env.PYTHON_API_URL ?? 'http://127.0.0.1:3001';

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

type ChartSpecs = Record<string, unknown>[] | null;

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

// The instance-wide toggle is resolved here, not trusted from the caller, so a
// stale tab can't opt back into charts once admins turn the setting off. Note
// it only gates chart *generation* — the result table still renders (and
// persists) when it is off.
const generateCharts = async (payload: Record<string, unknown>): Promise<ChartSpecs> => {
	if (!(await isVisualizationEnabled())) return null;

	const upstream = await fetch(`${PYTHON_API_URL}/api/chat/visualize`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(payload),
	});

	// Soft-fail: the chart is a nice-to-have, so fall through to "no chart"
	// rather than an error the UI would have to special-case.
	if (!upstream.ok) return null;
	const { charts } = (await upstream.json()) as { charts?: unknown };
	return Array.isArray(charts) && charts.length > 0
		? (charts as Record<string, unknown>[])
		: null;
};

const persistResultMessage = async (
	conversationId: string,
	result: unknown,
	charts: ChartSpecs,
): Promise<void> => {
	const message = buildResultMessage({ sql_response_from_db: result }, charts);
	if (!message) return;

	const prisma = getPrisma();
	await prisma.message.create({
		data: {
			conversation_id: conversationId,
			role: 'assistant',
			content: message.content,
			sql_code: null,
			sql_response: message.sqlResponse ?? null,
		},
	});
	await prisma.conversation.update({
		where: { id: conversationId },
		data: { updated_at: new Date() },
	});
};

// Same permission as the chat completions route: every chat user may trigger
// this, not just admins.
//
// Unlike the completions route, this one never creates a conversation: by the
// time step 2 runs, step 1 (or `POST /api/conversations`) has already created
// it. Falling back to create-on-missing here would leave an orphan
// conversation — just the chart/table bubble, with no question or Message 1
// ahead of it — whenever this is called with an id that never got that far
// (e.g. a standalone API caller skipping the completions step).
export const POST = withPermission({ chat: ['use'] })(async (req, { user }) => {
	const payload = parseBody(await req.text());
	const conversationId =
		typeof payload.conversation_id === 'string' && payload.conversation_id.trim()
			? payload.conversation_id
			: null;
	const conversation = conversationId ? await findOwnedConversation(user, conversationId) : null;
	const charts = await generateCharts(payload).catch(() => null);

	if (conversation) {
		try {
			await persistResultMessage(conversation.id, payload.result, charts);
		} catch {
			// Best-effort persistence — never fail the response over history.
		}
	}

	return NextResponse.json({ charts });
});
