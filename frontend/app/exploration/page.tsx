// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Suspense } from 'react';

import { ExplorationView } from '@/components/explorationPage';

export default function ExplorationPage() {
	return (
		<Suspense
			fallback={
				<div className="flex h-full items-center justify-center">
					<div
						className="h-10 w-10 animate-spin rounded-full border-2 border-zinc-200 border-t-[#76b900] dark:border-zinc-700"
						role="status"
						aria-label="Loading"
					/>
				</div>
			}
		>
			<ExplorationView />
		</Suspense>
	);
}
