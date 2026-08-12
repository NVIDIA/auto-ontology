// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Route Handler that proxies GET /api/chat/watch to the FastAPI backend,
// letting a reloaded/reopened browser tab reattach to a chat run already in
// progress for a conversation it owns. This is a passive observer only —
// it never submits a question and performs no persistence. The original
// FastAPI owns completion persistence regardless of who's watching, so a
// failed/dropped watch connection here has no effect on the underlying run or
// on what eventually gets saved.

import { withPermission } from '@/auth/with-auth';
import { findOwnedConversation } from '@/lib/chatConversations';
import { buildInternalIdentityHeaders } from '@/lib/internalIdentity';

const PYTHON_API_URL = process.env.PYTHON_API_URL ?? 'http://127.0.0.1:3001';

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

export const GET = withPermission({ conversation: ['read'] })(async (req, { user }) => {
	const conversationId = new URL(req.url).searchParams.get('conversation_id');
	if (!conversationId) {
		return new Response('Missing conversation_id', { status: 400 });
	}

	const conversation = await findOwnedConversation(user, conversationId);
	if (!conversation) {
		return new Response('Conversation not found', { status: 404 });
	}

	const upstream = await fetch(
		`${PYTHON_API_URL}/api/chat/watch?conversation_id=${encodeURIComponent(conversationId)}`,
		{
			headers: {
				Accept: 'text/event-stream',
				...buildInternalIdentityHeaders(user.id),
			},
		},
	);

	if (!upstream.ok || !upstream.body) {
		return new Response('Upstream watch request failed', {
			status: upstream.status || 502,
		});
	}

	return new Response(upstream.body, {
		status: 200,
		headers: {
			'Content-Type': 'text/event-stream; charset=utf-8',
			'Cache-Control': 'no-cache, no-transform',
			Connection: 'keep-alive',
			'X-Accel-Buffering': 'no',
		},
	});
});
