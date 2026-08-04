// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
	ChatRequest,
	ChatStreamEvent,
	StepEvent,
	ResultEvent,
	ErrorEvent,
	SqlResult,
	VisualizeRequest,
	VisualizeResponse,
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
 * `error` events to `callbacks` as they arrive. Shared by `streamChat`
 * (submitting a new question) and `watchChat` (reattaching to a run
 * already in progress) since both consume the exact same wire format.
 *
 * Resolves once `[DONE]` is seen. If the underlying connection closes
 * before `[DONE]` ever arrives — network drop, server restart, etc. — the
 * caller has no way to know how the run ended, so this surfaces it via
 * `onError` rather than leaving the UI stuck in a loading state forever.
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
						streamCompleted = true;
						callbacks.onResult(event);
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

	// Step 1 only — SQL + formatted answer. Charts are fetched separately via
	// `fetchCharts` once this stream's `result` event lands (a second step),
	// so this request never waits on chart generation.
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
 * Second step: once step 1 (`streamChat`) has returned the SQL and its
 * executed result, ask the server whether a chart applies. Resolves to
 * `null` on any failure or when visualization is disabled/skipped, so the
 * caller can always fall back to a plain table.
 *
 * `conversationId` lets the route persist the bubble this step produces —
 * the chart, or the table it falls back to — since the completions proxy
 * already wrote the prose bubble and cannot know how this resolves.
 */
export const fetchCharts = async (
	question: string,
	sql: string | undefined,
	result: SqlResult | undefined,
	conversationId: string | null,
): Promise<unknown> => {
	const payload: VisualizeRequest = {
		question,
		sql: sql ?? '',
		result,
		conversation_id: conversationId ?? undefined,
	};

	try {
		const res = await fetch('/api/chat/visualize', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(payload),
		});
		if (!res.ok) return null;

		const data = (await res.json()) as VisualizeResponse;
		return Array.isArray(data.charts) && data.charts.length > 0 ? data.charts : null;
	} catch {
		return null;
	}
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
			// Query param is deliberately `conversationId` (not `conversation_id`):
			// `proxy.ts` auto-rewrites any `/api/*` query key ending in `_id` into a
			// path segment (e.g. `?db_id=x` -> `/x`), which would 404 this route.
			const res = await fetch(
				`/api/chat/watch?conversationId=${encodeURIComponent(conversationId)}`,
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
	// Query param is deliberately `conversationId` — see `watchChat`.
	await fetch(`/api/chat/cancel?conversationId=${encodeURIComponent(conversationId)}`, {
		method: 'POST',
	});
};
