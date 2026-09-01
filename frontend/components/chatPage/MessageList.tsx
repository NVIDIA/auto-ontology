// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useEffect, useRef } from 'react';
import type { ChatMessage, GraphStep } from '@/types/chat';
import { EmptyState } from '@/common/EmptyState';
import { Icon, IconName } from '@/common/icons';
import { SkeletonBlock } from '@/common/Skeleton';
import { EmptyStateVariant } from '@/enums/emptyState';
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
			<div
				className="flex flex-1 flex-col gap-4 p-6"
				role="status"
				aria-label="Loading messages"
			>
				<SkeletonBlock className="h-16 w-2/3" />
				<SkeletonBlock className="ml-auto h-12 w-1/2" />
				<SkeletonBlock className="h-20 w-3/5" />
			</div>
		);
	}

	if (messages.length === 0 && !isLoading) {
		return (
			<EmptyState
				variant={EmptyStateVariant.Welcome}
				illustration={
					<div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-[#76b900]/15">
						<Icon name={IconName.ChatBubble} className="h-7 w-7 text-[#76b900]" />
					</div>
				}
				title="Ask a question"
				description="Type a natural-language question and the system will search your data, construct a SQL query, and return results."
			/>
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
