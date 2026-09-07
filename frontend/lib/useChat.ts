// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { cancelChat, streamChat, watchChat } from '@/api/chat';
import { conversationsApi, toConversation } from '@/api/conversations';
import { GENERIC_ANSWER_ERROR, buildSqlAnswerMessage, isResultMessage } from '@/lib/answerMessages';
import type { ChartsEvent, ChatMessage, GraphStep, ResultEvent, StepEvent } from '@/types/chat';

let nextId = 0;
const uid = () => `msg-${Date.now()}-${nextId++}`;

const CONVERSATION_IN_PROGRESS = 'Conversation in progress';

const REFETCH_DELAY_MS = 1000;
const REFETCH_ATTEMPTS = 3;

const sleep = (ms: number) =>
	new Promise<void>((resolve) => {
		setTimeout(resolve, ms);
	});

/**
 * Fold one step event into the list. A node sends two: `start`, which opens
 * the step and is what keeps the visible label honest while the node works,
 * and `end`, which only carries the thought the node produced — so `end`
 * updates the entry its `start` opened instead of adding a duplicate.
 *
 * An `end` with no matching open step (a backend that doesn't send `start`,
 * or a replay joined mid-node) falls through to appending, which is exactly
 * the behaviour before `phase` existed.
 */
const applyStepEvent = (prev: GraphStep[], event: StepEvent): GraphStep[] => {
	if (event.phase === 'end') {
		const open = prev[prev.length - 1];
		if (open?.node === event.node) {
			return [...prev.slice(0, -1), { ...open, thought: event.thought ?? open.thought }];
		}
	}

	return [
		...prev.map((step) => ({ ...step, status: 'completed' as const })),
		{
			node: event.node,
			label: event.label,
			thought: event.thought,
			status: 'active' as const,
		},
	];
};

export const useChat = () => {
	const [messages, setMessages] = useState<ChatMessage[]>([]);
	const [steps, setSteps] = useState<GraphStep[]>([]);
	// The validated query the running agent is executing, rendered inside the
	// thinking bubble so the user can read it while it runs instead of only
	// seeing it arrive with the answer. Replaced in place if execution fails
	// and the query is rebuilt; the answer bubble takes over once the run
	// resolves.
	const [liveSql, setLiveSql] = useState<string | null>(null);
	// Same value for the stream callbacks, which need it without re-creating
	// themselves on every SQL rewrite — `onError` reads it to keep the failed
	// query on screen.
	const liveSqlRef = useRef<string | null>(null);
	const [isLoading, setIsLoading] = useState(false);
	const controllerRef = useRef<AbortController | null>(null);
	// Bumped on every `sendMessage`/`stopGeneration`/`clearConversation` so a
	// `charts` event landing after the user stopped/started a new turn can
	// tell it's stale and skip mutating state.
	const requestIdRef = useRef(0);
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

	// Keeps the ref and the rendered value in step. `null` clears the block —
	// done at the start of every run so a new question never briefly shows the
	// previous one's SQL.
	const updateLiveSql = useCallback((sql: string | null) => {
		liveSqlRef.current = sql;
		setLiveSql(sql);
	}, []);

	// Purely local UI state — the chat proxy route persists both the user's
	// and the assistant's turns to the conversation server-side (so history
	// stays correct even if this component unmounts mid-stream), so there is
	// no DB write to do here.
	const appendAssistantMessage = useCallback(
		(content: string, extras?: { sql?: string; sqlResponse?: string; thoughts?: string }) => {
			const assistantMsg: ChatMessage = {
				id: uid(),
				role: 'assistant',
				content,
				sql: extras?.sql,
				sqlResponse: extras?.sqlResponse,
				thoughts: extras?.thoughts,
				timestamp: Date.now(),
			};
			setMessages((prev) => [...prev, assistantMsg]);
		},
		[],
	);

	// Message 1 — prose + SQL, rendered as soon as the SQL pipeline resolves.
	// FastAPI persists the same response + SQL fields, so a reloaded
	// conversation matches what the user watched arrive.
	const appendSqlAnswerMessage = useCallback(
		(answer: ResultEvent['answer']) => {
			const msg = buildSqlAnswerMessage(answer);
			if (msg) appendAssistantMessage(msg.content, { sql: msg.sql, thoughts: msg.thoughts });
		},
		[appendAssistantMessage],
	);

	// Message 2 — the chart, or the fallback result table, that the backend
	// already generated and persisted itself right after the SQL answer (see
	// `_pump` in gsf/server/chat/router.py). Rendering it is now just "append
	// what arrived on the wire" — no second request, and no "only the
	// submitting tab" caveat: it rides the same buffered stream `onResult`
	// did, so both a live send and a reattached `/chat/watch` get it exactly
	// the same way. `requestId` still guards against a stale event landing
	// after the user moved on to a different send/resume.
	const appendChartsMessage = useCallback(
		(event: ChartsEvent, requestId: number) => {
			if (requestId !== requestIdRef.current) return;
			appendAssistantMessage(event.content, { sqlResponse: event.sql_response ?? undefined });
		},
		[appendAssistantMessage],
	);

	// Picks up the assistant turn a run that finished just before we attached
	// left behind. Adopts every snapshot it reads, so the prose shows up as soon
	// as it is committed, but keeps polling until the turn is complete: Message 1
	// and Message 2 are two separate inserts a beat apart, so a transcript ending
	// in Message 1 may still be waiting on Message 2. Turns without an executed
	// result have no Message 2 at all, and simply run out of attempts with the
	// transcript already up to date.
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
					if (last?.role !== 'assistant') continue;

					setMessages(conv.messages);
					if (isResultMessage(last)) return;
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
		requestIdRef.current += 1;
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
			updateLiveSql(null);
			// Supersedes any charts event still pending from this tab's own
			// previous send, so its bubble can't land on top of this run.
			const requestId = (requestIdRef.current += 1);
			let sawActivity = false;

			const controller = watchChat(conversationId, {
				onStep(event) {
					sawActivity = true;
					activeRunConvIdRef.current = conversationId;
					setIsLoading(true);
					setSteps((prev) => applyStepEvent(prev, event));
				},

				// The buffered stream replays every event from the start, so a
				// tab that reattaches mid-run catches up to whatever SQL the
				// agent is on right now, exactly like one that watched all along.
				onSql(event) {
					updateLiveSql(event.sql);
				},

				// Message 1 lands here; Message 2 (chart/table) arrives as its own
				// `charts` event a little later on this same stream — the backend
				// generates and persists it itself now, so a run we merely
				// reattached to (e.g. navigated away mid-answer and came back)
				// gets it exactly like a live send would.
				//
				// The caller (ChatView) loads history *before* calling this, so
				// if Message 1 was already persisted by the time that fetch ran
				// — likely, since the slot now stays open through the chart step
				// — `messages` already ends with a dangling copy of this very
				// turn's reply (Message 1 alone, or Message 1 + Message 2). A
				// conversation only ever has one run in flight, so any trailing
				// assistant messages can only belong to the turn this replay is
				// about to rebuild from the live buffer: drop them first so it
				// doesn't land on top as a duplicate.
				onResult(event) {
					setMessages((prev) => {
						let cut = prev.length;
						while (cut > 0 && prev[cut - 1].role === 'assistant') cut -= 1;
						return cut === prev.length ? prev : prev.slice(0, cut);
					});
					appendSqlAnswerMessage(event.answer);
					// The answer bubble carries the SQL from here on; leaving the
					// live copy up would show it twice through the chart step.
					updateLiveSql(null);
					setSteps((prev) => prev.map((s) => ({ ...s, status: 'completed' as const })));

					if (!event.answer.sql_response_from_db) {
						if (resumeControllerRef.current === controller) {
							activeRunConvIdRef.current = null;
							resumeControllerRef.current = null;
						}
						setIsLoading(false);
						return;
					}

					setSteps((prev) => [
						...prev.map((s) => ({ ...s, status: 'completed' as const })),
						{ node: 'visualize', label: 'Building charts', status: 'active' as const },
					]);
				},

				onCharts(event) {
					appendChartsMessage(event, requestId);
					if (requestId === requestIdRef.current) {
						setSteps((prev) =>
							prev.map((s) => ({ ...s, status: 'completed' as const })),
						);
						setIsLoading(false);
					}
					if (resumeControllerRef.current === controller) {
						activeRunConvIdRef.current = null;
						resumeControllerRef.current = null;
					}
				},

				onError(event) {
					if (resumeControllerRef.current === controller) {
						activeRunConvIdRef.current = null;
						resumeControllerRef.current = null;
					}
					if (!sawActivity) {
						// The watch connection itself failed before we ever confirmed
						// a run was in progress — likely a network blip while probing
						// an idle conversation. Stay silent rather than injecting an
						// error bubble for something that may never have been running.
						return;
					}
					if (requestId !== requestIdRef.current) {
						return;
					}
					const message = event.message.trim() || GENERIC_ANSWER_ERROR;
					// Keep the query that failed attached to the error bubble —
					// the backend persists it the same way, so a reload shows the
					// same thing.
					appendAssistantMessage(message, { sql: liveSqlRef.current ?? undefined });
					updateLiveSql(null);
					setSteps((prev) => prev.map((s) => ({ ...s, status: 'completed' as const })));
					setIsLoading(false);
				},

				onDone() {
					if (resumeControllerRef.current === controller) {
						activeRunConvIdRef.current = null;
						resumeControllerRef.current = null;
					}
					if (sawActivity) {
						// Safety net: normally `onResult` (no executed result) or
						// `onCharts` already wrapped this up. Only matters if the
						// backend promised a `charts` event but the chart step then
						// failed to produce one — without this, the UI would be
						// stuck on "Building charts" forever.
						if (requestId === requestIdRef.current) {
							setSteps((prev) =>
								prev.map((s) => ({ ...s, status: 'completed' as const })),
							);
							setIsLoading(false);
						}
						return;
					}

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
		[
			setMessages,
			appendSqlAnswerMessage,
			appendAssistantMessage,
			appendChartsMessage,
			pollForPersistedAssistant,
			updateLiveSql,
		],
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
			updateLiveSql(null);
			setIsLoading(true);
			const requestId = (requestIdRef.current += 1);
			activeRunConvIdRef.current = conversationId;

			const finishRun = () => {
				setSteps((prev) => prev.map((s) => ({ ...s, status: 'completed' as const })));
				setIsLoading(false);
				activeRunConvIdRef.current = null;
				controllerRef.current = null;
			};

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
						setSteps((prev) => applyStepEvent(prev, event));
					},

					onSql(event) {
						updateLiveSql(event.sql);
					},

					onResult(event) {
						appendSqlAnswerMessage(event.answer);
						// The answer bubble owns the SQL from here on.
						updateLiveSql(null);

						if (!event.answer.sql_response_from_db) {
							// No executed result to visualize — the backend won't send a
							// `charts` event either, so wrap up here.
							finishRun();
							return;
						}

						// The backend is generating and persisting the chart/table
						// bubble itself now; keep the "thinking" indicator up under
						// its own label until `onCharts` delivers it on this stream.
						setSteps((prev) => [
							...prev.map((s) => ({ ...s, status: 'completed' as const })),
							{
								node: 'visualize',
								label: 'Building charts',
								status: 'active' as const,
							},
						]);
					},

					onCharts(event) {
						appendChartsMessage(event, requestId);
						if (requestId === requestIdRef.current) finishRun();
					},

					// Safety net: normally `onResult` (no executed result) or
					// `onCharts` already finished the run. This only matters if the
					// backend promised a `charts` event (an executed result) but the
					// chart step then failed to produce one — without this, the UI
					// would be stuck on "Building charts" forever.
					onDone() {
						if (requestId === requestIdRef.current) finishRun();
					},

					onError(event) {
						const message = event.message.trim() || GENERIC_ANSWER_ERROR;
						finishRun();

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

						// Keep the query that failed attached to the error bubble;
						// the backend persists it the same way, so a reload matches.
						appendAssistantMessage(message, {
							sql: liveSqlRef.current ?? undefined,
						});
						updateLiveSql(null);
						settle(true);
					},
				},
			);

			controllerRef.current = controller;
			return accepted;
		},
		[
			appendAssistantMessage,
			appendSqlAnswerMessage,
			appendChartsMessage,
			resumeIfRunning,
			updateLiveSql,
		],
	);

	// Leaves the run alone server-side: it keeps streaming into the buffer and
	// persists its answer, so reopening the conversation picks it back up.
	const clearConversation = useCallback(() => {
		stopGeneration(false);
		setMessages([]);
		setSteps([]);
		updateLiveSql(null);
	}, [stopGeneration, updateLiveSql]);

	return {
		messages,
		setMessages,
		steps,
		liveSql,
		isLoading,
		sendMessage,
		resumeIfRunning,
		stopGeneration,
		clearConversation,
	};
};
