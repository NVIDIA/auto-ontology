// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useEffect, useRef } from 'react';
import type { ChatMessage, GraphStep } from '@/types/chat';
import { Icon, IconName } from '@/common/icons';
import { MessageBubble } from './MessageBubble';
import { ThinkingMessage } from './ThinkingMessage';

type MessageListProps = {
	messages: ChatMessage[];
	isLoading: boolean;
	steps: GraphStep[];
	messageListLoading?: boolean;
};

export const MessageList = ({
	messages,
	isLoading,
	steps,
	messageListLoading = false,
}: MessageListProps) => {
	const bottomRef = useRef<HTMLDivElement>(null);

	useEffect(() => {
		bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
	}, [messages.length, isLoading, steps.length]);

	if (messageListLoading) {
		return (
			<div className="flex flex-1 items-center justify-center">
				<div
					className="h-10 w-10 animate-spin rounded-full border-2 border-zinc-200 border-t-[#76b900] dark:border-zinc-700"
					role="status"
					aria-label="Loading messages"
				/>
			</div>
		);
	}

	if (messages.length === 0 && !isLoading) {
		return (
			<div className="flex flex-1 flex-col items-center justify-center gap-3 px-4">
				<div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-[#76b900]/15">
					<Icon name={IconName.ChatBubble} className="h-7 w-7 text-[#76b900]" />
				</div>
				<h2 className="text-lg font-semibold text-zinc-800 dark:text-zinc-200">
					Ask a question
				</h2>
				<p className="max-w-sm text-center text-sm text-zinc-500 dark:text-zinc-400">
					Type a natural-language question and the system will search your data, construct
					a SQL query, and return results.
				</p>
			</div>
		);
	}

	return (
		<div className="flex-1 overflow-y-auto">
			<div className="mx-auto max-w-3xl space-y-4 px-4 py-6">
				{messages.map((msg) => (
					<MessageBubble key={msg.id} message={msg} />
				))}
				{isLoading && <ThinkingMessage steps={steps} />}
				<div ref={bottomRef} />
			</div>
		</div>
	);
};
