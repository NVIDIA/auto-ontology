// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useEffect, useState } from 'react';
import { usersApi } from '@/api/users';
import { Toast } from '@/common/Toast';

export default function AgentSettingsPage() {
	// Defaults to on (matches the `visualization` column's default) until the
	// server value is fetched, so the switch never flashes "off" first.
	const [visualization, setVisualization] = useState(true);
	const [saving, setSaving] = useState(false);
	const [error, setError] = useState<string | null>(null);

	useEffect(() => {
		let cancelled = false;
		usersApi.getVisualization().then((res) => {
			if (cancelled || res.error) return;
			setVisualization(res.visualization);
		});
		return () => {
			cancelled = true;
		};
	}, []);

	const handleToggle = async () => {
		if (saving) return;
		const next = !visualization;
		setSaving(true);
		setError(null);
		setVisualization(next);

		const res = await usersApi.setVisualization(next);
		setSaving(false);

		if (res.error) {
			setVisualization(!next); // roll back the optimistic update
			setError(res.message ?? 'Failed to update setting. Please try again.');
		}
	};

	return (
		<main className="min-h-0 min-w-0 flex-1 overflow-y-auto bg-[linear-gradient(180deg,rgba(255,255,255,1)_0%,rgba(250,250,250,0.6)_100%)] px-7 py-6 sm:px-10 sm:py-7 dark:bg-[linear-gradient(180deg,rgba(9,9,11,1)_0%,rgba(24,24,27,0.5)_100%)]">
			<div className="w-full space-y-5">
				<h1 className="text-base font-semibold text-zinc-900 dark:text-zinc-100">
					Agent Settings
				</h1>

				<div className="rounded-lg border border-zinc-200/90 bg-white/90 p-4 shadow-sm ring-1 ring-zinc-950/[0.04] dark:border-zinc-700/90 dark:bg-zinc-950/50 dark:ring-white/[0.06]">
					<div className="flex items-center justify-between gap-4">
						<div className="min-w-0">
							<h2 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
								Visualize SQL Results
							</h2>
							<p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
								Auto-generate visualizations for SQL results
							</p>
						</div>
						<button
							type="button"
							role="switch"
							aria-checked={visualization}
							aria-label="Visualize SQL Results"
							disabled={saving}
							onClick={() => {
								void handleToggle();
							}}
							className={`relative inline-flex h-6 w-11 shrink-0 cursor-pointer items-center rounded-full transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${
								visualization ? 'bg-[#76b900]' : 'bg-zinc-300 dark:bg-zinc-600'
							}`}
						>
							<span
								className={`inline-block h-5 w-5 transform rounded-full bg-white shadow transition-transform ${
									visualization ? 'translate-x-5' : 'translate-x-0.5'
								}`}
							/>
						</button>
					</div>
				</div>
			</div>

			<Toast
				open={error !== null}
				message={error ?? ''}
				variant="error"
				onClose={() => setError(null)}
			/>
		</main>
	);
}
