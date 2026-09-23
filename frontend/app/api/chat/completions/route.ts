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
// FastAPI owns completion persistence so web and Agent API callers share one
// history implementation. This route authenticates the caller, forwards its
// Auto Ontology user identity over the private upstream hop, and otherwise preserves SSE.

import { after } from 'next/server';
import { withPermission } from '@/auth/with-auth';
import { userCan } from '@/auth/permissions';
import { resolveSubjectToken } from '@/auth/sso-token';
import { buildInternalIdentityHeaders } from '@/lib/internalIdentity';

const PYTHON_API_URL = process.env.PYTHON_API_URL ?? 'http://127.0.0.1:3001';

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';
const SOURCE_HEADER = 'x-auto-ontology-source';

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

// Ask a question and stream the answer back as Server-Sent Events.
//
// FastAPI emits the SQL plus the formatted answer first (`result`), then —
// when that answer has an executed result — generates and persists the
// chart/table bubble itself and streams it back as its own `charts` event
// before the stream closes. No second request needed from this route or the
// browser. Passing `conversation_id` persists the turn and additionally
// requires `conversation: ['write']`; the route answers 409 while that
// conversation already has a run in flight. Omitting it runs the question
// statelessly, with no history and no chart step.
export const POST = withPermission({ chat: ['use'] })(async (req, { user }) => {
	const payload = parseBody(await req.text());
	const hasConversationId =
		typeof payload.conversation_id === 'string' && payload.conversation_id.trim() !== '';
	if (hasConversationId && !userCan(user, { conversation: ['write'] })) {
		return new Response('Forbidden', { status: 403 });
	}

	// Forwarded verbatim to FastAPI, which streams the SQL/answer and then the
	// chart/table bubble on this same connection — see the module docstring.
	const body = JSON.stringify(payload);

	// Forward the caller's SSO token whenever we have one. Only the backend knows
	// which connections are configured to authenticate as the signed-in user, so
	// it owns the fail-closed decision; sending the token is a no-op otherwise.
	const upstreamHeaders: Record<string, string> = {
		'Content-Type': 'application/json',
		Accept: 'text/event-stream',
		[SOURCE_HEADER]: req.headers.get(SOURCE_HEADER) ?? 'api',
		...buildInternalIdentityHeaders(user.id),
	};

	const subjectToken = await resolveSubjectToken(req.headers, user.id);
	if (subjectToken) upstreamHeaders.Authorization = `Bearer ${subjectToken}`;

	const upstream = await fetch(`${PYTHON_API_URL}/api/chat/completions`, {
		method: 'POST',
		headers: upstreamHeaders,
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

	const responseHeaders = {
		'Content-Type': 'text/event-stream; charset=utf-8',
		'Cache-Control': 'no-cache, no-transform',
		Connection: 'keep-alive',
		'X-Accel-Buffering': 'no',
	};

	const [toClient, toKeepAlive] = upstream.body.tee();
	after(async () => {
		try {
			await toKeepAlive.pipeTo(new WritableStream());
		} catch {
			// The browser-facing stream is already independent of this drain.
		}
	});
	return new Response(toClient, { status: 200, headers: responseHeaders });
});
