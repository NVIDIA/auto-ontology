// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Route Handler that proxies POST /api/chat/completions to the FastAPI
// backend and streams the SSE response straight back to the browser.
//
// We use a hand-rolled handler (instead of Next.js `rewrites`) because the
// dev-server rewrites buffer streaming responses — the browser would receive
// nothing until the upstream connection closed, defeating SSE.

import { after } from 'next/server';
import { withPermission } from '@/auth/with-auth';
import { getPrisma } from '@/lib/prisma';
import { resolveZoneIds } from '@/auth/resolve-zones';

const PYTHON_API_URL = process.env.PYTHON_API_URL ?? 'http://127.0.0.1:3001';

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';
const SOURCE_HEADER = 'x-gsf-source';

const extractQuestion = (rawBody: string): string => {
	try {
		const parsed = JSON.parse(rawBody) as { question?: unknown };
		return typeof parsed.question === 'string' ? parsed.question : '';
	} catch {
		return '';
	}
};

// Consume a teed copy of the SSE stream, returning the final answer's response
// text and SQL. Mirrors the parsing the browser does in `frontend/api/chat.ts`
// (data: lines, `[DONE]` sentinel, JSON `result`/`error` events).
const readFinalAnswer = async (
	stream: ReadableStream<Uint8Array>,
): Promise<{ response: string | null; sql: string | null }> => {
	const reader = stream.getReader();
	const decoder = new TextDecoder();
	let buffer = '';
	let response: string | null = null;
	let sql: string | null = null;

	const handleData = (data: string): void => {
		if (!data || data === '[DONE]') return;
		try {
			const event = JSON.parse(data) as {
				type?: string;
				answer?: { response?: string; sql_code?: string };
			};
			if (event.type === 'result') {
				response = event.answer?.response ?? null;
				sql = event.answer?.sql_code ?? null;
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

	return { response, sql };
};

export const POST = withPermission({ chat: ['use'] })(async (req, { user }) => {
	const rawBody = await req.text();

	// Resolve zone IDs server-side and inject them so the Python backend can
	// restrict retrieval candidates to zone-accessible tables.  The client only
	// sends { question }; zone_ids are added here to prevent spoofing and to
	// keep the client unaware of zone membership details.
	const zoneIds = await resolveZoneIds(user.id, user.role);

	// Viewer with no zone access: short-circuit before hitting the backend.
	// Without this guard, an empty zone list becomes `zone_ids: []` which the
	// Python router converts to None (falsy check), bypassing all zone filters
	// and effectively granting the viewer full catalog access — same bug that
	// proxyToBackend already guards against for catalog API routes.
	if (zoneIds !== null && zoneIds.length === 0) {
		return new Response(
			JSON.stringify({
				detail: "You don't have access to any data sources. Ask an admin to grant you zone access.",
			}),
			{ status: 403, headers: { 'Content-Type': 'application/json' } },
		);
	}

	let body: string;
	try {
		const parsed = JSON.parse(rawBody) as Record<string, unknown>;
		body = JSON.stringify({ ...parsed, zone_ids: zoneIds ?? [] });
	} catch {
		body = rawBody;
	}

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

	// Single writer for analytics, for every caller. Create the row up front,
	// then tee the stream — one branch flows to the client untouched, the other
	// is parsed after the response to backfill the final answer. `user` is the
	// resolved GSF user (session or SSO bearer), injected by withPermission.
	// question extraction runs on the original client body (before uid injection).
	const question = extractQuestion(rawBody);

	const prisma = getPrisma();
	const row = await prisma.conversationAnalytics.create({
		data: { question, source, userId: user.id },
	});

	const [toClient, toCapture] = upstream.body.tee();

	after(async () => {
		try {
			const { response, sql } = await readFinalAnswer(toCapture);
			await prisma.conversationAnalytics.update({
				where: { id: row.id },
				data: { response, sql, responseTimestamp: new Date() },
			});
		} catch {
			// Best-effort analytics — never let capture failures affect the
			// already-delivered chat response.
		}
	});

	return new Response(toClient, { status: 200, headers: responseHeaders });
});
