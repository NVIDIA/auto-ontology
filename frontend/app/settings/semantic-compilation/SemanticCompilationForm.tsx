// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useState } from 'react';
import { semanticCompilationApi } from '@/api/settings';
import { Toast } from '@/common/Toast';

export const SemanticCompilationForm = ({ initialEnabled }: { initialEnabled: boolean }) => {
	// Seeded from the server (see page.tsx) so the correct state renders on first
	// paint; updated optimistically and rolled back if the save fails.
	const [enabled, setEnabled] = useState(initialEnabled);
	const [saving, setSaving] = useState(false);
	const [error, setError] = useState<string | null>(null);
	const [message, setMessage] = useState<string | null>(null);

	const handleToggle = async () => {
		if (saving) return;
		const next = !enabled;
		setSaving(true);
		setError(null);
		setMessage(null);
		setEnabled(next);

		try {
			const result = await semanticCompilationApi.setEnabled(next);
			setEnabled(result.enabled);
			setMessage(
				result.enabled
					? 'Semantic compilation enabled — a compilation run has been triggered.'
					: 'Semantic compilation disabled.',
			);
		} catch {
			setEnabled(!next); // roll back the optimistic update
			setError('Failed to update semantic compilation. Please try again.');
		} finally {
			setSaving(false);
		}
	};

	return (
		<div className="h-full overflow-auto p-6">
			<div className="mx-auto max-w-xl">
				<h1 className="mb-1 text-lg font-semibold text-zinc-900 dark:text-zinc-100">
					Semantic Compilation
				</h1>
				<p className="mb-4 text-xs text-zinc-500">
					When enabled, the ingestion service compiles the semantic layer for every
					connected database on startup and once every 24 hours. Enabling it also triggers
					a compilation run immediately.
				</p>

				<div className="flex items-center justify-between rounded-lg border border-zinc-200 px-4 py-3 dark:border-zinc-700">
					<div className="flex flex-col">
						<span className="text-sm font-medium text-zinc-900 dark:text-zinc-100">
							Enable semantic compilation
						</span>
						<span className="text-xs text-zinc-500">
							{enabled ? 'Running on the 24h schedule.' : 'Currently off.'}
						</span>
					</div>
					<button
						type="button"
						role="switch"
						aria-checked={enabled}
						aria-label="Enable semantic compilation"
						disabled={saving}
						onClick={handleToggle}
						className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer items-center rounded-full transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${
							enabled ? 'bg-[#76b900]' : 'bg-zinc-300 dark:bg-zinc-600'
						}`}
					>
						<span
							className={`inline-block h-5 w-5 transform rounded-full bg-white shadow transition-transform ${
								enabled ? 'translate-x-5' : 'translate-x-0.5'
							}`}
						/>
					</button>
				</div>
			</div>

			<Toast
				open={error !== null}
				message={error ?? ''}
				variant="error"
				onClose={() => setError(null)}
			/>
			<Toast
				open={message !== null}
				message={message ?? ''}
				variant="success"
				onClose={() => setMessage(null)}
			/>
		</div>
	);
};
