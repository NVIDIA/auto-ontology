// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { Spinner } from '@nvidia/foundations-react-core';
import { usePathname, useRouter, useSearchParams } from 'next/navigation';
import { useChat } from '@/lib/useChat';
import type { Conversation } from '@/types/chat';
import {
	conversationsApi,
	toConversation,
	type ConversationSummary,
	type ConversationDetail,
} from '@/api/conversations';
import { semanticCompilationApi } from '@/api/settings';
import { ChatSidebar } from './ChatSidebar';
import { MessageList } from './MessageList';
import { ChatInput } from './ChatInput';
import { SemanticNotReady } from './SemanticNotReady';

export const ChatView = () => {
	const router = useRouter();
	const pathname = usePathname();
	const searchParams = useSearchParams();
	const rawFocus = searchParams.get('focus');
	const focusId = rawFocus != null && rawFocus.trim() !== '' ? rawFocus.trim() : null;

	const [activeConvId, setActiveConvId] = useState<string | null>(focusId);
	const [sidebarOpen, setSidebarOpen] = useState(false);
	const [conversations, setConversations] = useState<Conversation[]>([]);
	const [sidebarLoading, setSidebarLoading] = useState(true);
	const [messageListLoading, setMessageListLoading] = useState<boolean>(focusId != null);
	// null = still checking; false = semantic layer missing (block the chat area).
	const [semanticReady, setSemanticReady] = useState<boolean | null>(null);
	const loadedFocusRef = useRef<string | null>(null);

	const {
		messages,
		setMessages,
		steps,
		liveSql,
		isLoading,
		sendMessage,
		resumeIfRunning,
		stopGeneration,
		clearConversation,
	} = useChat();

	const updateFocusInUrl = useCallback(
		(id: string | null) => {
			const params = new URLSearchParams(searchParams.toString());
			if (id) {
				params.set('focus', id);
			} else {
				params.delete('focus');
			}
			const query = params.toString();
			router.replace(query ? `${pathname}?${query}` : pathname, { scroll: false });
		},
		[router, pathname, searchParams],
	);

	const refreshConversations = useCallback(async () => {
		try {
			const summaries = await conversationsApi.list();
			setConversations(
				summaries.map((s: ConversationSummary) => ({
					id: s.id,
					title: s.title || 'New conversation',
					messages: [],
					createdAt: new Date(s.created_at).getTime(),
				})),
			);
		} catch {
			// ignore fetch errors
		}
	}, []);

	useEffect(() => {
		setSidebarLoading(true);
		refreshConversations().finally(() => setSidebarLoading(false));

		semanticCompilationApi
			.getStatus()
			.then((res) => setSemanticReady(res.calculated))
			.catch(() => setSemanticReady(false));
	}, [refreshConversations]);

	useEffect(() => {
		if (!focusId) {
			loadedFocusRef.current = null;
			setMessageListLoading(false);
			return;
		}
		if (loadedFocusRef.current === focusId) return;
		loadedFocusRef.current = focusId;
		clearConversation();

		let active = true;
		setMessageListLoading(true);
		conversationsApi
			.get(focusId)
			.then((detail: ConversationDetail) => {
				if (!active) return;
				const conv = toConversation(detail);
				setActiveConvId(detail.id);
				setMessages(conv.messages);
				resumeIfRunning(detail.id);
			})
			.catch(() => {
				if (!active) return;
				loadedFocusRef.current = null;
				updateFocusInUrl(null);
			})
			.finally(() => {
				if (active) setMessageListLoading(false);
			});
		return () => {
			active = false;
		};
	}, [focusId, setMessages, resumeIfRunning, clearConversation, updateFocusInUrl]);

	const handleNewChat = useCallback(async () => {
		clearConversation();
		setActiveConvId(null);
		setSidebarOpen(false);
		loadedFocusRef.current = null;
		updateFocusInUrl(null);
		await refreshConversations();
	}, [clearConversation, refreshConversations, updateFocusInUrl]);

	const handleSelectConversation = useCallback(
		async (id: string) => {
			if (id === activeConvId) {
				setSidebarOpen(false);
				updateFocusInUrl(id);
				return;
			}

			clearConversation();
			setMessageListLoading(true);
			try {
				const detail: ConversationDetail = await conversationsApi.get(id);
				const conv = toConversation(detail);
				loadedFocusRef.current = id;
				setActiveConvId(id);
				setMessages(conv.messages);
				resumeIfRunning(id);
				setSidebarOpen(false);
				updateFocusInUrl(id);
			} catch {
				// ignore fetch errors
			} finally {
				setMessageListLoading(false);
			}
		},
		[activeConvId, setMessages, resumeIfRunning, clearConversation, updateFocusInUrl],
	);

	const handleRename = useCallback(
		async (id: string, title: string) => {
			try {
				await conversationsApi.rename(id, title);
				await refreshConversations();
			} catch {
				// ignore
			}
		},
		[refreshConversations],
	);

	const handleDelete = useCallback(
		async (id: string) => {
			try {
				await conversationsApi.delete(id);
				if (activeConvId === id) {
					clearConversation();
					setActiveConvId(null);
					loadedFocusRef.current = null;
					updateFocusInUrl(null);
				}
				await refreshConversations();
			} catch {
				// ignore
			}
		},
		[activeConvId, clearConversation, refreshConversations, updateFocusInUrl],
	);

	const handleSend = useCallback(
		async (text: string): Promise<boolean> => {
			let convId = activeConvId;
			if (!convId) {
				try {
					const title = text.slice(0, 50) || 'New conversation';
					const created = await conversationsApi.create(title);
					convId = created.id;
					loadedFocusRef.current = convId;
					setActiveConvId(convId);
					updateFocusInUrl(convId);
					refreshConversations();
				} catch {
					return false;
				}
			}
			return sendMessage(text, convId);
		},
		[activeConvId, sendMessage, refreshConversations, updateFocusInUrl],
	);

	return (
		<div className="flex h-full bg-white dark:bg-zinc-950">
			<ChatSidebar
				conversations={conversations}
				activeId={activeConvId}
				onSelect={handleSelectConversation}
				onNewChat={handleNewChat}
				onRename={handleRename}
				onDelete={handleDelete}
				isOpen={sidebarOpen}
				onToggle={() => setSidebarOpen((o) => !o)}
				sidebarLoading={sidebarLoading}
			/>

			<main className="flex min-w-0 flex-1 flex-col">
				{semanticReady == null ? (
					<div className="flex flex-1 items-center justify-center">
						<Spinner aria-label="Checking semantic layer" className="h-10 w-10" />
					</div>
				) : !semanticReady ? (
					<SemanticNotReady />
				) : (
					<>
						<MessageList
							messages={messages}
							isLoading={isLoading}
							steps={steps}
							liveSql={liveSql}
							messageListLoading={messageListLoading}
						/>

						<ChatInput
							onSend={handleSend}
							onStop={stopGeneration}
							isLoading={isLoading}
						/>
					</>
				)}
			</main>
		</div>
	);
};
