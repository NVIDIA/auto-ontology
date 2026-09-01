// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Suspense } from 'react';
import { SkeletonBlock, SkeletonCard, SkeletonRows } from '@/common/Skeleton';
import { DataWorkspaceView } from '@/components/dataPage';

export default function DataPage() {
	return (
		<Suspense
			fallback={
				<div className="flex h-screen" role="status" aria-label="Loading data workspace">
					<aside className="w-72 border-r border-zinc-200 p-4 dark:border-zinc-800">
						<SkeletonBlock className="mb-5 h-6 w-24" />
						<SkeletonRows rows={8} />
					</aside>
					<main className="flex flex-1 flex-col gap-4 p-6">
						<SkeletonCard rows={4} />
						<SkeletonCard rows={4} />
					</main>
				</div>
			}
		>
			<DataWorkspaceView />
		</Suspense>
	);
}
