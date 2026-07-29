// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
	ChatRequest,
	ChatStreamEvent,
	StepEvent,
	ResultEvent,
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

export type StreamChatCallbacks = {
	/**
	 * Fires once the backend has accepted the request (HTTP 200 + body ready
	 * to stream). Useful for committing optimistic state — e.g. the user
	 * message — only after the server agreed to process it, so failures
	 * before the stream starts (e.g. 409 "Conversation in progress") don't
	 * leave orphan messages in the conversation.
	 */
	onStart?: () => void;
	onStep: (event: StepEvent) => void;
	onResult: (event: ResultEvent) => void;
	onError: (event: ErrorEvent) => void;
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

	// The visualization flag is resolved server-side from the instance-wide
	// Agent Settings configuration, so it is deliberately not sent here.
	const body = JSON.stringify({ question: payload.question });

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

			const reader = res.body.getReader();
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
						return;
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
			}

			// Stream closed without [DONE]/result/error — surface as an error
			// so the UI can drop out of loading instead of hanging forever.
			if (!streamCompleted) {
				callbacks.onError({
					type: 'error',
					message: 'Connection closed before the agent finished.',
				});
			}
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
