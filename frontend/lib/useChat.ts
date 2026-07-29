// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { streamChat } from '@/api/chat';
import { stringifySqlResponse } from '@/lib/sqlResponse';
import type { ChatMessage, GraphStep } from '@/types/chat';

let nextId = 0;
const uid = () => `msg-${Date.now()}-${nextId++}`;

const GENERIC_ERROR_MESSAGE =
	'Something went wrong. Please try again, and if the issue persists, contact our support';

export const useChat = () => {
	const [messages, setMessages] = useState<ChatMessage[]>([]);
	const [steps, setSteps] = useState<GraphStep[]>([]);
	const [isLoading, setIsLoading] = useState(false);
	const controllerRef = useRef<AbortController | null>(null);

	// Purely local UI state — the chat proxy route persists both the user's
	// and the assistant's turns to the conversation server-side (so history
	// stays correct even if this component unmounts mid-stream), so there is
	// no DB write to do here.
	const appendAssistantMessage = useCallback(
		(content: string, extras?: { sql?: string; sqlResponse?: string }) => {
			const assistantMsg: ChatMessage = {
				id: uid(),
				role: 'assistant',
				content,
				sql: extras?.sql,
				sqlResponse: extras?.sqlResponse,
				timestamp: Date.now(),
			};
			setMessages((prev) => [...prev, assistantMsg]);
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
			// the conversation yet. We only show it once the backend accepts the
			// request via the `onStart` callback below; on 409 "Conversation in
			// progress" (or any other pre-stream error) it never appears, keeping
			// the visible chat history in sync with what the proxy route actually
			// persisted (or didn't).
			setSteps([]);
			setIsLoading(true);

			const controller = streamChat(
				{ question: text, conversationId },
				{
					onStart() {
						setMessages((prev) => [...prev, userMsg]);
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

						appendAssistantMessage(response, {
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
						appendAssistantMessage(message);
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
