// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Suspense } from 'react';
import { Icon, IconName } from '@/common/icons';
import { TermsLoadingSkeleton, TermsView } from '@/components/termsPage';

export default function TermsPage() {
	return (
		<Suspense
			fallback={
				<div className="flex h-full w-full flex-col bg-white dark:bg-zinc-950">
					<header className="flex items-center gap-3 border-b border-zinc-200 px-6 py-4 dark:border-zinc-800">
						<Icon name={IconName.Terms} className="h-5 w-5 text-[#76b900]" />
						<h1 className="text-lg font-semibold tracking-tight text-zinc-900 dark:text-zinc-100">
							Terms
						</h1>
					</header>
					<div className="flex-1 px-6 py-6">
						<TermsLoadingSkeleton />
					</div>
				</div>
			}
		>
			<TermsView />
		</Suspense>
	);
}
