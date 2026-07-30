// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { cancelChat, streamChat, watchChat } from '@/api/chat';
import { conversationsApi, toConversation } from '@/api/conversations';
import { buildAnswerMessages } from '@/lib/answerMessages';
import type { ChatMessage, GraphStep, ResultEvent } from '@/types/chat';

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
	// The conversation whose run this hook is currently attached to, so Stop
	// can cancel it server-side without the caller having to pass it back in.
	const activeRunConvIdRef = useRef<string | null>(null);
	// In-flight cancel from Stop — the next send awaits this so it doesn't race
	// the server still holding the conversation slot (and get a 409).
	const cancelInFlightRef = useRef<Promise<void> | null>(null);
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

	// Renders one agent answer the same way the completions proxy persists it,
	// so a reloaded conversation matches what the user watched arrive. Shared
	// by the live send and the resume watch.
	const appendAnswer = useCallback(
		(answer: ResultEvent['answer']) => {
			buildAnswerMessages(answer).forEach((msg) => {
				appendAssistantMessage(msg.content, {
					sql: msg.sql,
					sqlResponse: msg.sqlResponse,
				});
			});
		},
		[appendAssistantMessage],
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

	// Detaches this tab from the run. `cancel` additionally aborts the run
	// server-side, which is what the Stop button wants: the agent holds the
	// conversation's slot until it finishes, so merely dropping the stream
	// would leave the next question rejected with 409.
	const stopGeneration = useCallback((cancel = true) => {
		controllerRef.current?.abort();
		controllerRef.current = null;
		resumeControllerRef.current?.abort();
		resumeControllerRef.current = null;

		const conversationId = activeRunConvIdRef.current;
		activeRunConvIdRef.current = null;
		if (cancel && conversationId) {
			const pending = cancelChat(conversationId).catch(() => {});
			cancelInFlightRef.current = pending;
			void pending.finally(() => {
				if (cancelInFlightRef.current === pending) {
					cancelInFlightRef.current = null;
				}
			});
		}

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
					activeRunConvIdRef.current = conversationId;
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
					appendAnswer(event.answer);
					setSteps((prev) => prev.map((s) => ({ ...s, status: 'completed' as const })));
					setIsLoading(false);
					activeRunConvIdRef.current = null;
					resumeControllerRef.current = null;
				},

				onError(event) {
					activeRunConvIdRef.current = null;
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
					activeRunConvIdRef.current = null;
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
		[appendAnswer, appendAssistantMessage, pollForPersistedAssistant],
	);

	// Resolves true once the backend accepted the question, false if it was
	// refused before the stream started — the caller uses that to decide
	// whether the typed text can be discarded.
	const sendMessage = useCallback(
		async (text: string, conversationId: string | null): Promise<boolean> => {
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

			// Show the user turn immediately — waiting for `onStart` made the
			// bubble lag behind a post-cancel worker cold-start, so the input
			// looked empty for a noticeable beat. On a pre-stream refusal we
			// roll it back so the transcript still matches what was persisted.
			setMessages((prev) => [...prev, userMsg]);
			setSteps([]);
			setIsLoading(true);
			activeRunConvIdRef.current = conversationId;

			// Wait out a Stop's cancel so we don't POST into a slot that's still
			// held for a few hundred ms (which would 409 and leave the question
			// looking like it never sent).
			const pendingCancel = cancelInFlightRef.current;
			if (pendingCancel) await pendingCancel;

			let settle: (accepted: boolean) => void = () => {};
			const accepted = new Promise<boolean>((resolve) => {
				settle = resolve;
			});

			const controller = streamChat(
				{ question: text, conversationId },
				{
					onStart() {
						settle(true);
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
						appendAnswer(event.answer);
						setSteps((prev) =>
							prev.map((s) => ({ ...s, status: 'completed' as const })),
						);
						setIsLoading(false);
						activeRunConvIdRef.current = null;
						controllerRef.current = null;
					},

					onError(event) {
						const message = event.message.trim() || GENERIC_ERROR_MESSAGE;
						setSteps((prev) =>
							prev.map((s) => ({ ...s, status: 'completed' as const })),
						);
						setIsLoading(false);
						activeRunConvIdRef.current = null;
						controllerRef.current = null;

						// Another stream already owns this conversation (a second tab,
						// or a run we stopped watching but couldn't cancel). Nothing
						// was submitted, so roll back the optimistic user turn, keep
						// the question in the input, and attach to the live run.
						if (message === CONVERSATION_IN_PROGRESS && conversationId) {
							setMessages((prev) => prev.filter((m) => m.id !== userMsg.id));
							settle(false);
							resumeIfRunning(conversationId);
							return;
						}

						appendAssistantMessage(message);
						settle(true);
					},
				},
			);

			controllerRef.current = controller;
			return accepted;
		},
		[appendAnswer, appendAssistantMessage, resumeIfRunning],
	);

	// Leaves the run alone server-side: it keeps streaming into the buffer and
	// persists its answer, so reopening the conversation picks it back up.
	const clearConversation = useCallback(() => {
		stopGeneration(false);
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
