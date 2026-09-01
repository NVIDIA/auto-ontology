// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
	ChatRequest,
	ChatStreamEvent,
	StepEvent,
	ResultEvent,
	ChartsEvent,
	ErrorEvent,
} from '@/types/chat';

const getResponseErrorMessage = async (res: Response): Promise<string> => {
	const fallback = `Server responded with ${res.status}`;
	const text = await res.text().catch(() => '');
	if (!text) return fallback;

	try {
		const { detail } = JSON.parse(text) as { detail?: unknown };
		if (typeof detail === 'string' && detail.trim()) return detail;
	} catch {}
	return text;
};

export type ChatEventCallbacks = {
	onStep: (event: StepEvent) => void;
	onResult: (event: ResultEvent) => void;
	onCharts?: (event: ChartsEvent) => void;
	onError: (event: ErrorEvent) => void;
	onDone?: () => void;
};

export type StreamChatCallbacks = ChatEventCallbacks & {
	/**
	 * Fires once the backend has accepted the request (HTTP 200 + body ready
	 * to stream). Useful for committing optimistic state — e.g. the user
	 * message — only after the server agreed to process it, so failures
	 * before the stream starts (e.g. 409 "Conversation in progress") don't
	 * leave orphan messages in the conversation.
	 */
	onStart?: () => void;
};

/**
 * Reads an already-open SSE `Response` body, dispatching `step`/`result`/
 * `charts`/`error` events to `callbacks` as they arrive. Shared by
 * `streamChat` (submitting a new question) and `watchChat` (reattaching to a
 * run already in progress) since both consume the exact same wire format.
 *
 * Resolves once `[DONE]` is seen — which, unlike before, is now the *only*
 * normal terminator: `result` no longer ends the read early, because the
 * backend keeps the stream open a little longer to also generate and persist
 * the chart/table bubble (`charts`), and this needs to stay listening to
 * receive it. If the underlying connection closes before `[DONE]` ever
 * arrives — network drop, server restart, etc. — the caller has no way to
 * know how the run ended, so this surfaces it via `onError` rather than
 * leaving the UI stuck in a loading state forever.
 */
const consumeSseStream = async (res: Response, callbacks: ChatEventCallbacks): Promise<void> => {
	const reader = res.body!.getReader();
	const decoder = new TextDecoder();
	let buffer = '';
	let streamCompleted = false;

	while (true) {
		const { done, value } = await reader.read();
		if (done) break;

		buffer += decoder.decode(value, { stream: true });
		const lines = buffer.split('\n');
		buffer = lines.pop() ?? '';

		for (const line of lines) {
			const trimmed = line.trim();
			if (!trimmed.startsWith('data: ')) continue;

			const data = trimmed.slice(6);
			if (data === '[DONE]') {
				streamCompleted = true;
				break;
			}

			try {
				const event: ChatStreamEvent = JSON.parse(data);
				switch (event.type) {
					case 'step':
						callbacks.onStep(event);
						break;
					case 'result':
						callbacks.onResult(event);
						break;
					case 'charts':
						callbacks.onCharts?.(event);
						break;
					case 'error':
						streamCompleted = true;
						callbacks.onError(event);
						break;
				}
			} catch {
				// skip malformed lines
			}
		}
		if (streamCompleted) break;
	}

	// Stream closed without [DONE]/result/error — surface as an error so
	// the UI can drop out of loading instead of hanging forever.
	if (!streamCompleted) {
		callbacks.onError({
			type: 'error',
			message: 'Connection closed before the agent finished.',
		});
		return;
	}

	callbacks.onDone?.();
};

/**
 * Opens an SSE connection to POST /api/chat/completions and invokes
 * callbacks as events arrive. Returns an AbortController the caller
 * can use to cancel mid-stream.
 */
export const streamChat = (
	payload: ChatRequest,
	callbacks: StreamChatCallbacks,
): AbortController => {
	const controller = new AbortController();

	// The backend answers with the SQL/prose first (`result`), then generates
	// and persists the chart/table bubble itself and streams it back as a
	// separate `charts` event — no second request needed from here.
	const body = JSON.stringify({
		question: payload.question,
		conversation_id: payload.conversationId ?? undefined,
		target_db: payload.target_db ?? undefined,
	});

	(async () => {
		try {
			const res = await fetch('/api/chat/completions', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json', 'x-gsf-source': 'app' },
				body,
				signal: controller.signal,
			});

			if (!res.ok || !res.body) {
				callbacks.onError({
					type: 'error',
					message: await getResponseErrorMessage(res),
				});
				return;
			}

			callbacks.onStart?.();
			await consumeSseStream(res, callbacks);
		} catch (err: unknown) {
			if (err instanceof DOMException && err.name === 'AbortError') return;
			callbacks.onError({
				type: 'error',
				message: err instanceof Error ? err.message : 'Unknown error',
			});
		}
	})();

	return controller;
};

/**
 * Reattaches to a run already in progress for `conversationId`, if any —
 * e.g. after a page reload or reopening the browser mid-response. Replays
 * every step buffered server-side since the run started, then keeps
 * tailing live updates the same way `streamChat` does.
 *
 * If nothing is running for this conversation, the connection closes
 * immediately with `[DONE]` and `onDone` fires with no step/result/error
 * events — callers should treat that as a silent no-op, not an error.
 */
export const watchChat = (
	conversationId: string,
	callbacks: ChatEventCallbacks,
): AbortController => {
	const controller = new AbortController();

	(async () => {
		try {
			const res = await fetch(
				`/api/chat/watch?conversation_id=${encodeURIComponent(conversationId)}`,
				{ signal: controller.signal },
			);

			if (!res.ok || !res.body) {
				callbacks.onDone?.();
				return;
			}

			await consumeSseStream(res, callbacks);
		} catch (err: unknown) {
			if (err instanceof DOMException && err.name === 'AbortError') return;
			callbacks.onError({
				type: 'error',
				message: err instanceof Error ? err.message : 'Unknown error',
			});
		}
	})();

	return controller;
};

/**
 * Aborts the run in flight for `conversationId`, freeing the conversation
 * so the next question isn't rejected with 409. Best-effort: aborting the
 * browser's own stream is what the user sees, and a failed cancel only
 * means the agent finishes on its own.
 */
export const cancelChat = async (conversationId: string): Promise<void> => {
	await fetch(`/api/chat/cancel?conversation_id=${encodeURIComponent(conversationId)}`, {
		method: 'POST',
	});
};
