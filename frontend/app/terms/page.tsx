// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Suspense } from 'react';
import { TermsView } from '@/components/termsPage';

export default function TermsPage() {
	return (
		<Suspense
			fallback={
				<div className="flex h-screen items-center justify-center">
					<div
						className="h-10 w-10 animate-spin rounded-full border-2 border-zinc-200 border-t-[#76b900] dark:border-zinc-700"
						role="status"
						aria-label="Loading"
					/>
				</div>
			}
		>
			<TermsView />
		</Suspense>
	);
}
