// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import Link from 'next/link';
import { Icon, IconName } from '@/components/icons';

// Blocks the chat conversation area when the semantic layer hasn't been
// calculated yet: the agent can't answer questions without it.
export const SemanticNotReady = () => (
	<div className="flex flex-1 items-center justify-center p-6">
		<div className="max-w-md text-center">
			<div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-[#76b900]/15">
				<Icon name={IconName.NvidiaLogo} className="h-6 w-6 text-[#76b900]" />
			</div>
			<h2 className="mb-2 text-lg font-semibold text-zinc-900 dark:text-zinc-100">
				Semantic layer not ready
			</h2>
			<p className="mb-4 text-sm text-zinc-500 dark:text-zinc-400">
				The semantic layer hasn&apos;t been created yet, so I can&apos;t answer questions.
			</p>
			<Link
				href="/settings/semantic-compilation"
				className="inline-flex items-center rounded-lg bg-[#76b900] px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-[#6aa600]"
			>
				Enable semantic compilation
			</Link>
		</div>
	</div>
);
