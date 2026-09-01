// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Suspense } from 'react';

import { SkeletonBlock, SkeletonRows } from '@/common/Skeleton';
import { ChatView } from '@/components/chatPage';
import { SkeletonVariant } from '@/enums/skeleton';

// ChatView reads `useSearchParams()`, so during static prerender Next.js
// requires the subtree to be wrapped in a Suspense boundary. The fallback
// renders the chat shell background so we don't get a layout shift while
// the client portion hydrates.
const ChatFallback = () => (
	<div className="flex h-full bg-white dark:bg-zinc-950" aria-hidden>
		<aside className="hidden w-[296px] border-r border-zinc-200 p-3 dark:border-zinc-800 lg:block">
			<SkeletonBlock className="mb-6 h-8 w-24" />
			<SkeletonBlock variant={SkeletonVariant.RECTANGLE} className="mb-4 h-10 w-full" />
			<SkeletonRows rows={8} />
		</aside>
		<main className="flex flex-1 flex-col justify-between p-6">
			<div className="space-y-4">
				<SkeletonBlock className="h-16 w-2/3" />
				<SkeletonBlock className="ml-auto h-12 w-1/2" />
				<SkeletonBlock className="h-20 w-3/5" />
			</div>
			<SkeletonBlock variant={SkeletonVariant.RECTANGLE} className="h-12 w-full" />
		</main>
	</div>
);

export default function ChatPage() {
	return (
		<Suspense fallback={<ChatFallback />}>
			<ChatView />
		</Suspense>
	);
}
