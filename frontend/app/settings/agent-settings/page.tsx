// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useEffect, useState } from 'react';
import { sqlQueryTimeoutApi, visualizationApi } from '@/api/settings';
import { Button } from '@/common/Button';
import { Toast } from '@/common/Toast';
import { ButtonTheme, Size } from '@/enums/button';
import { ToastVariant } from '@/enums/toast';
import {
	DEFAULT_SQL_QUERY_TIMEOUT_SECONDS,
	MAX_SQL_QUERY_TIMEOUT_SECONDS,
	MIN_SQL_QUERY_TIMEOUT_SECONDS,
	isValidSqlQueryTimeout,
} from '@/lib/sqlQueryTimeout';

export default function AgentSettingsPage() {
	// Defaults to on (matches a missing `visualization_enabled` configuration)
	// until the server value is fetched, so the switch never flashes "off" first.
	const [visualization, setVisualization] = useState(true);
	const [saving, setSaving] = useState(false);
	const [error, setError] = useState<string | null>(null);

	// `timeoutSeconds` is what the server holds; `timeoutDraft` is the raw input text.
	const [timeoutSeconds, setTimeoutSeconds] = useState(DEFAULT_SQL_QUERY_TIMEOUT_SECONDS);
	const [timeoutDraft, setTimeoutDraft] = useState(String(DEFAULT_SQL_QUERY_TIMEOUT_SECONDS));
	const [savingTimeout, setSavingTimeout] = useState(false);

	const draftSeconds = Number(timeoutDraft);
	const draftValid = timeoutDraft.trim() !== '' && isValidSqlQueryTimeout(draftSeconds);
	const timeoutDirty = draftValid && draftSeconds !== timeoutSeconds;

	useEffect(() => {
		let cancelled = false;
		visualizationApi.get().then((res) => {
			if (cancelled || res.error) return;
			setVisualization(res.enabled);
		});
		sqlQueryTimeoutApi.get().then((res) => {
			if (cancelled || res.error) return;
			setTimeoutSeconds(res.seconds);
			setTimeoutDraft(String(res.seconds));
		});
		return () => {
			cancelled = true;
		};
	}, []);

	const handleSaveTimeout = async () => {
		if (savingTimeout || !timeoutDirty) return;
		setSavingTimeout(true);
		setError(null);

		const res = await sqlQueryTimeoutApi.set(draftSeconds);
		setSavingTimeout(false);

		if (res.error) {
			setError(res.message ?? 'Failed to update setting. Please try again.');
			return;
		}
		setTimeoutSeconds(res.seconds);
		setTimeoutDraft(String(res.seconds));
	};

	const handleToggle = async () => {
		if (saving) return;
		const next = !visualization;
		setSaving(true);
		setError(null);
		setVisualization(next);

		const res = await visualizationApi.setEnabled(next);
		setSaving(false);

		if (res.error) {
			setVisualization(!next); // roll back the optimistic update
			setError(res.message ?? 'Failed to update setting. Please try again.');
		}
	};

	return (
		<main className="min-h-0 min-w-0 flex-1 overflow-y-auto bg-[linear-gradient(180deg,rgba(255,255,255,1)_0%,rgba(250,250,250,0.6)_100%)] px-7 py-6 sm:px-10 sm:py-7 dark:bg-[linear-gradient(180deg,rgba(9,9,11,1)_0%,rgba(24,24,27,0.5)_100%)]">
			<div className="w-full space-y-5">
				<h1 className="text-base font-semibold text-heading dark:text-zinc-100">
					Agent Settings
				</h1>

				<div className="rounded-lg border border-zinc-200/90 bg-white/90 p-4 shadow-sm ring-1 ring-zinc-950/[0.04] dark:border-zinc-700/90 dark:bg-zinc-950/50 dark:ring-white/[0.06]">
					<div className="flex items-center justify-between gap-4">
						<div className="min-w-0">
							<h2 className="text-sm font-semibold text-heading dark:text-zinc-100">
								Visualize SQL Results
							</h2>
							<p className="mt-1 text-xs text-secondary dark:text-zinc-400">
								Auto-generate visualizations for SQL results, for all users
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

				<div className="rounded-lg border border-zinc-200/90 bg-white/90 p-4 shadow-sm ring-1 ring-zinc-950/[0.04] dark:border-zinc-700/90 dark:bg-zinc-950/50 dark:ring-white/[0.06]">
					<form
						className="flex items-center justify-between gap-4"
						onSubmit={(e) => {
							e.preventDefault();
							void handleSaveTimeout();
						}}
					>
						<div className="min-w-0">
							<label
								htmlFor="sql-query-timeout"
								className="text-sm font-semibold text-heading dark:text-zinc-100"
							>
								SQL Query Timeout
							</label>
							<p className="mt-1 text-xs text-secondary dark:text-zinc-400">
								Seconds a query generated by the agent may run before it is
								cancelled ({MIN_SQL_QUERY_TIMEOUT_SECONDS}–
								{MAX_SQL_QUERY_TIMEOUT_SECONDS}, default{' '}
								{DEFAULT_SQL_QUERY_TIMEOUT_SECONDS})
							</p>
						</div>
						<div className="flex shrink-0 items-center gap-2">
							<input
								id="sql-query-timeout"
								type="number"
								inputMode="numeric"
								min={MIN_SQL_QUERY_TIMEOUT_SECONDS}
								max={MAX_SQL_QUERY_TIMEOUT_SECONDS}
								step={1}
								value={timeoutDraft}
								disabled={savingTimeout}
								aria-invalid={!draftValid}
								onChange={(e) => setTimeoutDraft(e.target.value)}
								className="h-9 w-24 rounded-md border border-zinc-300 bg-white px-3 text-sm text-body outline-none transition-colors focus:border-[#76b900] focus:ring-2 focus:ring-[#76b900]/30 disabled:cursor-not-allowed disabled:bg-zinc-100 disabled:text-disabled aria-[invalid=true]:border-red-500 dark:border-zinc-600 dark:bg-zinc-900 dark:text-zinc-300 dark:disabled:bg-zinc-800/50 dark:disabled:text-zinc-400"
							/>
							<Button
								theme={ButtonTheme.Primary}
								size={Size.REGULAR}
								type="submit"
								disabled={savingTimeout || !timeoutDirty}
							>
								{savingTimeout ? 'Saving…' : 'Save'}
							</Button>
						</div>
					</form>
				</div>
			</div>

			<Toast
				open={error !== null}
				message={error ?? ''}
				variant={ToastVariant.Error}
				onClose={() => setError(null)}
			/>
		</main>
	);
}
