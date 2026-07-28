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

const GENERIC_ERROR_MESSAGE =
	'Something went wrong. Please try again, and if the issue persists, contact our support';

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
	const controllerRef = useRef<AbortController | null>(null);
	// Tracks whether `onStart` fired for the in-flight request, so `onError`
	// can tell a pre-stream rejection (user turn never committed) apart from
	// a mid-stream failure (user turn already committed and persisted).
	const startedRef = useRef(false);

	const appendAssistantMessage = useCallback(
		(
			conversationId: string | null,
			content: string,
			extras?: { sql?: string; sqlResponse?: string },
			// False for a pre-stream failure (e.g. 409 "Conversation in progress"):
			// the matching user turn was never persisted either, so persisting only
			// this half would leave an orphan assistant row in the conversation.
			persist = true,
		) => {
			const assistantMsg: ChatMessage = {
				id: uid(),
				role: 'assistant',
				content,
				sql: extras?.sql,
				sqlResponse: extras?.sqlResponse,
				timestamp: Date.now(),
			};
			setMessages((prev) => [...prev, assistantMsg]);

			if (!conversationId || !persist) return;

			// Persist the assistant turn to the conversation history. Analytics
			// is captured server-side in the chat proxy route, so there is no
			// analytics work to do here.
			conversationsApi
				.addMessage(conversationId, {
					role: 'assistant',
					content,
					sqlCode: extras?.sql ?? null,
					sqlResponse: extras?.sqlResponse ?? null,
				})
				.catch(() => {});
		},
		[],
	);

	useEffect(
		() => () => {
			controllerRef.current?.abort();
			controllerRef.current = null;
		},
		[],
	);

	const sendMessage = useCallback(
		(text: string, conversationId: string | null) => {
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
			startedRef.current = false;

			const controller = streamChat(
				{ question: text, conversationId },
				{
					onStart() {
						startedRef.current = true;
						setMessages((prev) => [...prev, userMsg]);
						if (conversationId) {
							conversationsApi
								.addMessage(conversationId, { role: 'user', content: text })
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
						const {
							response,
							sql_code: sqlCode,
							sql_response_from_db: sqlResponseFromDb,
						} = event.answer;
						const sqlResponse = stringifySqlResponse(sqlResponseFromDb);

						appendAssistantMessage(conversationId, response, {
							sql: sqlCode,
							sqlResponse,
						});
						setSteps((prev) =>
							prev.map((s) => ({ ...s, status: 'completed' as const })),
						);
						setIsLoading(false);
						controllerRef.current = null;
					},

					onError(event) {
						const message = event.message.trim() || GENERIC_ERROR_MESSAGE;
						appendAssistantMessage(
							conversationId,
							message,
							undefined,
							startedRef.current,
						);
						setSteps((prev) =>
							prev.map((s) => ({ ...s, status: 'completed' as const })),
						);
						setIsLoading(false);
						controllerRef.current = null;
					},
				},
			);

			controllerRef.current = controller;
		},
		[appendAssistantMessage],
	);

	const clearConversation = useCallback(() => {
		controllerRef.current?.abort();
		controllerRef.current = null;
		setMessages([]);
		setSteps([]);
		setIsLoading(false);
	}, []);

	return {
		messages,
		setMessages,
		steps,
		isLoading,
		sendMessage,
		clearConversation,
	};
};
