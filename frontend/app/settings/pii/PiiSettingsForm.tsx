// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useState } from 'react';
import { piiDetectionApi } from '@/api/settings';
import { Toast } from '@/common/Toast';
import { Toggle } from '@/common/Toggle';
import { ToastVariant } from '@/enums/toast';

export const PiiSettingsForm = ({ initialEnabled }: { initialEnabled: boolean }) => {
	// Seeded from the server (see page.tsx) so the correct state renders on first
	// paint; updated optimistically and rolled back if the save fails.
	const [enabled, setEnabled] = useState(initialEnabled);
	const [saving, setSaving] = useState(false);
	const [error, setError] = useState<string | null>(null);
	const [message, setMessage] = useState<string | null>(null);

	const handleToggle = async (next: boolean) => {
		if (saving || next === enabled) return;
		setSaving(true);
		setError(null);
		setMessage(null);
		setEnabled(next);

		try {
			const result = await piiDetectionApi.setEnabled(next);
			setEnabled(result.enabled);
			setMessage(
				result.enabled
					? 'PII detection enabled — applies from the next ingest.'
					: 'PII detection disabled — ingest will skip PII classification.',
			);
		} catch {
			setEnabled(!next); // roll back the optimistic update
			setError('Failed to update PII detection. Please try again.');
		} finally {
			setSaving(false);
		}
	};

	return (
		<div className="h-full overflow-auto p-6">
			<div className="mx-auto max-w-xl">
				<h1 className="mb-1 text-lg font-semibold text-heading dark:text-zinc-100">
					PII Settings
				</h1>
				<p className="mb-4 text-xs text-secondary">
					When enabled, ingest classifies catalog columns and attaches the shared PII tag.
					Turn it off to skip that step; existing tags are left in place.
				</p>

				<div className="flex items-center justify-between rounded-lg border border-zinc-200 px-4 py-3 dark:border-zinc-700">
					<div className="flex flex-col pr-4">
						<span className="text-sm font-medium text-heading dark:text-zinc-100">
							PII Detection
						</span>
						<span className="text-xs text-secondary">
							{enabled
								? 'On — ingest classifies columns and tags PII.'
								: 'Off — ingest skips PII classification.'}
						</span>
					</div>
					<Toggle
						checked={enabled}
						aria-label="PII Detection"
						disabled={saving}
						onChange={handleToggle}
					/>
				</div>
			</div>

			<Toast
				open={error !== null}
				message={error ?? ''}
				variant={ToastVariant.Error}
				onClose={() => setError(null)}
			/>
			<Toast
				open={message !== null}
				message={message ?? ''}
				variant={ToastVariant.Success}
				onClose={() => setMessage(null)}
			/>
		</div>
	);
};
