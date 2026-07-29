// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { streamChat, watchChat } from '@/api/chat';
import { conversationsApi, toConversation } from '@/api/conversations';
import { stringifySqlResponse } from '@/lib/sqlResponse';
import type { ChatMessage, GraphStep } from '@/types/chat';

let nextId = 0;
const uid = () => `msg-${Date.now()}-${nextId++}`;

const GENERIC_ERROR_MESSAGE =
	'Something went wrong. Please try again, and if the issue persists, contact our support';

const CONVERSATION_IN_PROGRESS = 'Conversation in progress';

const REFETCH_DELAY_MS = 1000;
const REFETCH_ATTEMPTS = 3;

const sleep = (ms: number) =>
	new Promise<void>((resolve) => {
		setTimeout(resolve, ms);
	});

export const useChat = () => {
	const [messages, setMessages] = useState<ChatMessage[]>([]);
	const [steps, setSteps] = useState<GraphStep[]>([]);
	const [isLoading, setIsLoading] = useState(false);
	const controllerRef = useRef<AbortController | null>(null);
	// Tracks a `resumeIfRunning` watch, separately from `controllerRef` (the
	// user's own in-flight send) so switching conversations or sending a new
	// message never gets tangled up with a background resume attempt.
	const resumeControllerRef = useRef<AbortController | null>(null);
	// Mirrors `messages` for the async watch callbacks, which need the current
	// transcript without re-creating `resumeIfRunning` on every new message.
	const messagesRef = useRef(messages);
	useEffect(() => {
		messagesRef.current = messages;
	}, [messages]);

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

	const pollForPersistedAssistant = useCallback(
		async (conversationId: string, watchController: AbortController) => {
			for (let attempt = 0; attempt < REFETCH_ATTEMPTS; attempt += 1) {
				if (watchController.signal.aborted || controllerRef.current) return;
				await sleep(REFETCH_DELAY_MS);
				if (watchController.signal.aborted || controllerRef.current) return;

				try {
					const detail = await conversationsApi.get(conversationId);
					const conv = toConversation(detail);
					const last = conv.messages[conv.messages.length - 1];
					if (last?.role === 'assistant') {
						setMessages(conv.messages);
						return;
					}
				} catch {
					// ignore fetch errors and retry
				}
			}
		},
		[],
	);

	useEffect(
		() => () => {
			controllerRef.current?.abort();
			controllerRef.current = null;
			resumeControllerRef.current?.abort();
			resumeControllerRef.current = null;
		},
		[],
	);

	const stopGeneration = useCallback(() => {
		controllerRef.current?.abort();
		controllerRef.current = null;
		resumeControllerRef.current?.abort();
		resumeControllerRef.current = null;
		setSteps((prev) => prev.map((s) => ({ ...s, status: 'completed' as const })));
		setIsLoading(false);
	}, []);

	// Reattaches to a run already in progress for `conversationId`, if any —
	// e.g. after a page reload or reopening the browser mid-response. The
	// backend replays every step buffered since the run started, so this
	// rebuilds `steps`/`isLoading` exactly as if this tab had been watching
	// the whole time, then appends the final answer once it lands. If
	// nothing is running, `watchChat` closes immediately with no events and
	// this is a silent no-op — no loading flash, no stray messages.
	const resumeIfRunning = useCallback(
		(conversationId: string) => {
			// Never disturb an active local send.
			if (controllerRef.current) return;

			resumeControllerRef.current?.abort();
			setSteps([]);
			let sawActivity = false;

			const controller = watchChat(conversationId, {
				onStep(event) {
					sawActivity = true;
					setIsLoading(true);
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

					appendAssistantMessage(response, { sql: sqlCode, sqlResponse });
					setSteps((prev) => prev.map((s) => ({ ...s, status: 'completed' as const })));
					setIsLoading(false);
					resumeControllerRef.current = null;
				},

				onError(event) {
					if (!sawActivity) {
						// The watch connection itself failed before we ever confirmed
						// a run was in progress — likely a network blip while probing
						// an idle conversation. Stay silent rather than injecting an
						// error bubble for something that may never have been running.
						resumeControllerRef.current = null;
						return;
					}
					const message = event.message.trim() || GENERIC_ERROR_MESSAGE;
					appendAssistantMessage(message);
					setSteps((prev) => prev.map((s) => ({ ...s, status: 'completed' as const })));
					setIsLoading(false);
					resumeControllerRef.current = null;
				},

				onDone() {
					if (sawActivity) {
						resumeControllerRef.current = null;
						return;
					}

					resumeControllerRef.current = null;

					// Watch closed with nothing to replay. If the loaded history ends
					// on a user turn, the run most likely just finished and the
					// completions route's `after()` hook hasn't committed the assistant
					// row yet — poll briefly before giving up.
					const last = messagesRef.current[messagesRef.current.length - 1];
					if (last?.role === 'user') {
						void pollForPersistedAssistant(conversationId, controller);
					}
				},
			});

			resumeControllerRef.current = controller;
		},
		[appendAssistantMessage, pollForPersistedAssistant],
	);

	const sendMessage = useCallback(
		(text: string, conversationId: string | null) => {
			// A background resume may still be tailing this conversation (its
			// `isLoading` only flips on the first step, so the input isn't locked).
			// Drop it — this send supersedes it.
			resumeControllerRef.current?.abort();
			resumeControllerRef.current = null;

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

						if (message === CONVERSATION_IN_PROGRESS && conversationId) {
							setSteps((prev) =>
								prev.map((s) => ({ ...s, status: 'completed' as const })),
							);
							setIsLoading(false);
							controllerRef.current = null;
							resumeIfRunning(conversationId);
							return;
						}

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
		[appendAssistantMessage, resumeIfRunning],
	);

	const clearConversation = useCallback(() => {
		stopGeneration();
		setMessages([]);
		setSteps([]);
	}, [stopGeneration]);

	return {
		messages,
		setMessages,
		steps,
		isLoading,
		sendMessage,
		resumeIfRunning,
		stopGeneration,
		clearConversation,
	};
};
