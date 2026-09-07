// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Suspense } from 'react';
import { AnalysisView } from '@/components/analysisPage';
import { SkeletonCard } from '@/common/Skeleton';

export default function AnalysisPage() {
	return (
		<Suspense
			fallback={
				<div className="flex h-full w-full flex-col bg-white dark:bg-zinc-950">
					<div className="flex-1 px-6 py-6">
						<SkeletonCard rows={4} />
					</div>
				</div>
			}
		>
			<AnalysisView />
		</Suspense>
	);
}
