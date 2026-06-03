// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { streamChat } from '@/api/chat';
import { conversationsApi } from '@/api/conversations';
import type { ChatMessage, GraphStep } from '@/types/chat';

let nextId = 0;
const uid = () => `msg-${Date.now()}-${nextId++}`;

// How long an error toast stays on screen before auto-clearing. After this
// the user can retry sending — the next /api/chat/completions call naturally
// re-checks server availability (lock state) on the backend.
const ERROR_AUTO_DISMISS_MS = 4000;

// The agent returns the executed-DB rows under `sql_response_from_db`. It can
// be a stringified markdown/CSV table or a structured ``list[dict]`` payload.
// Normalise both shapes into a single string so DB persistence and parsing in
// `DynamicTable` stay simple (compact JSON for objects → cheap to re-parse).
const stringifySqlResponse = (value: unknown): string | undefined => {
	if (value == null) return undefined;
	if (typeof value === 'string') return value.trim() ? value : undefined;
	try {
		return JSON.stringify(value);
	} catch {
		return undefined;
	}
};

export const useChat = () => {
	const [messages, setMessages] = useState<ChatMessage[]>([]);
	const [steps, setSteps] = useState<GraphStep[]>([]);
	const [isLoading, setIsLoading] = useState(false);
	const [error, setError] = useState<string | null>(null);
	const controllerRef = useRef<AbortController | null>(null);
	const errorTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

	const clearErrorTimer = useCallback(() => {
		if (errorTimerRef.current !== null) {
			clearTimeout(errorTimerRef.current);
			errorTimerRef.current = null;
		}
	}, []);

	const showError = useCallback(
		(message: string) => {
			clearErrorTimer();
			setError(message);
			errorTimerRef.current = setTimeout(() => {
				setError(null);
				errorTimerRef.current = null;
			}, ERROR_AUTO_DISMISS_MS);
		},
		[clearErrorTimer],
	);

	// Abort any in-flight stream and clear the error timer when the consumer
	// unmounts (e.g. user navigates away from the chat page mid-"thinking").
	// Without this the SSE connection keeps the backend conversation lock
	// held until the response arrives — and then is silently discarded.
	useEffect(
		() => () => {
			clearErrorTimer();
			controllerRef.current?.abort();
			controllerRef.current = null;
		},
		[clearErrorTimer],
	);

	const sendMessage = useCallback(
		(text: string, convId: string | null) => {
			const userMsg: ChatMessage = {
				id: uid(),
				role: 'user',
				content: text,
				timestamp: Date.now(),
			};

			// Lock the input immediately so the Send button morphs into Stop and
			// duplicate sends are ignored — but DO NOT add the user message to
			// the conversation yet. We only commit it (UI + DB) once the backend
			// accepts the request via the `onStart` callback below; on 409
			// "Conversation in progress" (or any other pre-stream error) the
			// message is never persisted, keeping the chat history clean.
			setSteps([]);
			setIsLoading(true);
			clearErrorTimer();
			setError(null);

			const controller = streamChat(
				{ question: text },
				{
					onStart() {
						setMessages((prev) => [...prev, userMsg]);
						if (convId) {
							conversationsApi
								.addMessage(convId, { role: 'user', content: text })
								.catch(() => {});
						}
					},

					onStep(event) {
						setSteps((prev) => {
							const completed = prev.map((s) => ({
								...s,
								status: 'completed' as const,
							}));
							return [
								...completed,
								{ node: event.node, label: event.label, status: 'active' },
							];
						});
					},

					onResult(event) {
						const answer = event.answer;
						const content =
							typeof answer.response === 'string'
								? answer.response
								: JSON.stringify(answer, null, 2);
						const sql =
							typeof answer.sql_code === 'string' ? answer.sql_code : undefined;
						const sqlResponse = stringifySqlResponse(answer.sql_response_from_db);

						const assistantMsg: ChatMessage = {
							id: uid(),
							role: 'assistant',
							content,
							sql,
							sqlResponse,
							timestamp: Date.now(),
						};

						setMessages((prev) => [...prev, assistantMsg]);
						setSteps((prev) =>
							prev.map((s) => ({ ...s, status: 'completed' as const })),
						);
						setIsLoading(false);
						controllerRef.current = null;

						if (convId) {
							conversationsApi
								.addMessage(convId, {
									role: 'assistant',
									content,
									sqlCode: sql ?? null,
									sqlResponse: sqlResponse ?? null,
								})
								.catch(() => {});
						}
					},

					onError(event) {
						showError(event.message);
						setIsLoading(false);
						controllerRef.current = null;
					},
				},
			);

			controllerRef.current = controller;
		},
		[clearErrorTimer, showError],
	);

	const stopGeneration = useCallback(() => {
		controllerRef.current?.abort();
		controllerRef.current = null;
		setIsLoading(false);
	}, []);

	const clearMessages = useCallback(() => {
		controllerRef.current?.abort();
		controllerRef.current = null;
		clearErrorTimer();
		setMessages([]);
		setSteps([]);
		setIsLoading(false);
		setError(null);
	}, [clearErrorTimer]);

	const clearError = useCallback(() => {
		clearErrorTimer();
		setError(null);
	}, [clearErrorTimer]);

	return {
		messages,
		setMessages,
		steps,
		isLoading,
		error,
		sendMessage,
		stopGeneration,
		clearMessages,
		clearError,
	};
};
