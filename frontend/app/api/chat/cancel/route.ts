// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Route Handler that proxies POST /api/chat/cancel to the FastAPI backend,
// aborting the agent run in flight for a conversation the caller owns. This
// is what makes the chat UI's Stop button actually stop the run instead of
// merely detaching the browser from it — leaving the run alive would keep
// the conversation's slot claimed and reject the next question with 409.
//
// Cancelling ends the run without a final answer, so FastAPI leaves the
// persisted user turn without an assistant reply.

import { withPermission } from '@/auth/with-auth';
import { findOwnedConversation } from '@/lib/chatConversations';
import { buildInternalIdentityHeaders } from '@/lib/internalIdentity';

const PYTHON_API_URL = process.env.PYTHON_API_URL ?? 'http://127.0.0.1:3001';

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

// Abort the agent run in flight for a conversation the caller owns.
//
// This is what makes the chat UI's Stop button actually stop the run rather than
// merely detach the browser from it: an abandoned run keeps the conversation's
// slot claimed and the next question is rejected with 409. Cancelling ends the
// run without a final answer, so the persisted user turn keeps no assistant
// reply. 404 when the conversation does not exist or belongs to someone else.
export const POST = withPermission({ conversation: ['write'] })(async (req, { user }) => {
	const conversationId = new URL(req.url).searchParams.get('conversation_id');
	const conversation = conversationId ? await findOwnedConversation(user, conversationId) : null;

	if (!conversation) {
		return new Response('Conversation not found', { status: 404 });
	}

	const upstream = await fetch(
		`${PYTHON_API_URL}/api/chat/cancel?conversation_id=${encodeURIComponent(conversation.id)}`,
		{
			method: 'POST',
			headers: buildInternalIdentityHeaders(user.id),
		},
	);

	if (!upstream.ok) {
		return new Response('Upstream cancel request failed', {
			status: upstream.status || 502,
		});
	}

	return new Response(await upstream.text(), {
		status: 200,
		headers: { 'Content-Type': 'application/json' },
	});
});
