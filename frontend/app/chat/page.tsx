// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Suspense } from 'react';

import { ChatView } from '@/components/chatPage';

// ChatView reads `useSearchParams()`, so during static prerender Next.js
// requires the subtree to be wrapped in a Suspense boundary. The fallback
// renders the chat shell background so we don't get a layout shift while
// the client portion hydrates.
const ChatFallback = () => <div className="h-full bg-white dark:bg-zinc-950" aria-hidden />;

export default function ChatPage() {
	return (
		<Suspense fallback={<ChatFallback />}>
			<ChatView />
		</Suspense>
	);
}
