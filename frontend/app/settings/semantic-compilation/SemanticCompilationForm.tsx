// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useEffect, useState } from 'react';
import { columnProfilingApi, semanticCompilationApi } from '@/api/settings';
import { formatDate } from '@/common/date';
import { Icon, IconName } from '@/common/icons';
import { ConfirmModal } from '@/common/modal';
import { Spinner } from '@/common/Spinner';
import { Toast } from '@/common/Toast';
import { Toggle } from '@/common/Toggle';
import { ToastVariant } from '@/enums/toast';

export const SemanticCompilationForm = ({
	initialEnabled,
	initialProfilingEnabled,
	hasDatabases,
}: {
	initialEnabled: boolean;
	initialProfilingEnabled: boolean;
	hasDatabases: boolean;
}) => {
	// Seeded from the server (see page.tsx) so the correct state renders on first
	// paint; updated optimistically and rolled back if the save fails.
	const [enabled, setEnabled] = useState(initialEnabled);
	const [profilingEnabled, setProfilingEnabled] = useState(initialProfilingEnabled);
	const [savingProfiling, setSavingProfiling] = useState(false);
	const [saving, setSaving] = useState(false);
	const [confirmModalOpen, setConfirmModalOpen] = useState(false);
	const [error, setError] = useState<string | null>(null);
	const [message, setMessage] = useState<string | null>(null);
	const [running, setRunning] = useState(false);
	const [lastSuccessAt, setLastSuccessAt] = useState<string | null>(null);
	const [lastFailureAt, setLastFailureAt] = useState<string | null>(null);
	useEffect(() => {
		semanticCompilationApi
			.getStatus()
			.then((result) => {
				setRunning(result.running);
				setLastSuccessAt(result.last_success_at);
				setLastFailureAt(result.last_failure_at);
			})
			.catch(() => {});
	}, []);
	// Only set when it's more recent than the last success (see history.py),
	// so a stale failure never outlives the run that fixed it.
	const failed = !running && lastFailureAt !== null;
	// Whether semantic_compilation_history has anything to report yet. A newly
	// enabled instance that hasn't finished its first pass has none of these,
	// so there's nothing meaningful to show — surfacing "ready" with a "—"
	// placeholder would be misleading.
	const hasHistory = running || lastSuccessAt !== null || lastFailureAt !== null;

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
			if (result.enabled) {
				// Enabling triggers a run immediately (see the PUT route), so by the
				// time this resolves the ingestion service has already inserted a
				// history row and is very likely already running — refresh status
				// so the section appears now instead of only on the next page load.
				semanticCompilationApi
					.getStatus()
					.then((status) => {
						setRunning(status.running);
						setLastSuccessAt(status.last_success_at);
						setLastFailureAt(status.last_failure_at);
					})
					.catch(() => {});
			}
		} catch {
			setEnabled(!next); // roll back the optimistic update
			setError('Failed to update semantic compilation. Please try again.');
		} finally {
			setSaving(false);
		}
	};

	const handleProfilingToggle = async () => {
		if (savingProfiling) return;
		const next = !profilingEnabled;
		setSavingProfiling(true);
		setError(null);
		setMessage(null);
		setProfilingEnabled(next);

		try {
			const result = await columnProfilingApi.setEnabled(next);
			setProfilingEnabled(result.enabled);
			// No run is triggered: the flag is read at the start of each run, so
			// say when it takes effect rather than implying something happened now.
			setMessage(
				result.enabled
					? 'Column value profiling enabled — applies from the next compilation run.'
					: 'Column value profiling disabled — applies from the next compilation run.',
			);
		} catch {
			setProfilingEnabled(!next); // roll back the optimistic update
			setError('Failed to update column value profiling. Please try again.');
		} finally {
			setSavingProfiling(false);
		}
	};

	const handleReset = () => {
		setError(null);
		setConfirmModalOpen(false);
		// The ingestion service deletes in the background and answers 202, so the
		// request only starts the reset. Nothing here depends on the response, so
		// it is left unawaited and only a failure to reach the API is surfaced.
		setMessage('Semantic layer reset started — it runs in the background.');
		semanticCompilationApi.reset().catch(() => {
			setMessage(null);
			setError('Failed to reset the semantic layer. Please try again.');
		});
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

				{!hasDatabases && (
					<div className="mb-3 flex items-center rounded-lg border border-zinc-200 px-4 py-3 dark:border-zinc-700">
						<span className="text-sm font-medium text-zinc-900 dark:text-zinc-100">
							Please connect any data source
						</span>
					</div>
				)}

				{hasDatabases && (
					<>
						{enabled && hasHistory && (
							<div className="mb-3 flex items-center justify-between rounded-lg border border-zinc-200 px-4 py-3 dark:border-zinc-700">
								<div className="flex flex-col">
									<span className="text-sm font-medium text-zinc-900 dark:text-zinc-100">
										{running && 'Semantic compilation is running'}
										{!running && failed && 'Semantic compilation failed'}
										{!running && !failed && 'Semantic compilation is ready'}
									</span>
									<span className="text-xs text-zinc-500">
										{failed && !running
											? `Last attempt: ${formatDate(lastFailureAt as string, 'MMM DD YYYY, HH:mm Z')}`
											: `Last semantic compilation: ${
													lastSuccessAt
														? formatDate(
																lastSuccessAt,
																'MMM DD YYYY, HH:mm Z',
															)
														: '—'
												}`}
									</span>
								</div>
								{running ? (
									<span title="Semantic compilation is running">
										<Spinner className="h-4 w-4 shrink-0 text-[#76b900]" />
									</span>
								) : failed ? (
									<span
										className="flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-red-500/15 text-red-600 dark:text-red-400"
										title="Semantic compilation failed"
									>
										<Icon name={IconName.Close} className="h-3 w-3" />
									</span>
								) : (
									<span
										className="flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-[#76b900]/15 text-[#76b900]"
										title="Semantic compilation is ready"
									>
										<Icon name={IconName.Check} className="h-3 w-3" />
									</span>
								)}
							</div>
						)}

						{!enabled && (
							<div className="mb-3 flex items-center rounded-lg border border-zinc-200 px-4 py-3 dark:border-zinc-700">
								<span className="text-sm font-medium text-zinc-900 dark:text-zinc-100">
									You can enable the semantic compilation
								</span>
							</div>
						)}

						{/* Above the compilation toggle on purpose: this is a property of
						    how compilation runs, so it can be set before turning
						    compilation on, and stays editable afterwards. */}
						<div className="mb-3 flex items-center justify-between rounded-lg border border-zinc-200 px-4 py-3 dark:border-zinc-700">
							<div className="flex flex-col pr-4">
								<span className="text-sm font-medium text-zinc-900 dark:text-zinc-100">
									Profiling Column Values
								</span>
								<span className="text-xs text-zinc-500">
									{profilingEnabled
										? 'Samples live values from each table to capture example values, uniqueness and date formats.'
										: 'Off — the semantic layer is built from names and types only.'}
								</span>
							</div>
							<Toggle
								checked={profilingEnabled}
								aria-label="Profiling Column Values"
								disabled={savingProfiling}
								onChange={handleProfilingToggle}
							/>
						</div>

						<div className="flex items-center justify-between rounded-lg border border-zinc-200 px-4 py-3 dark:border-zinc-700">
							<div className="flex flex-col">
								<span className="text-sm font-medium text-zinc-900 dark:text-zinc-100">
									Enable semantic compilation
								</span>
								<span className="text-xs text-zinc-500">
									{enabled ? 'Running on the 24h schedule.' : 'Currently off.'}
								</span>
							</div>
							<Toggle
								checked={enabled}
								aria-label="Enable semantic compilation"
								disabled={saving}
								onChange={handleToggle}
							/>
						</div>

						{enabled && (
							<div className="mt-3 flex items-center justify-between rounded-lg border border-zinc-200 px-4 py-3 dark:border-zinc-700">
								<div className="flex flex-col">
									<span className="text-sm font-medium text-zinc-900 dark:text-zinc-100">
										Reset semantic layer
									</span>
									<span className="text-xs text-zinc-500">
										Deletes the semantic layer for every database, and rebuild
										it from scratch.
									</span>
								</div>
								<button
									type="button"
									disabled={saving}
									onClick={() => setConfirmModalOpen(true)}
									className="cursor-pointer rounded-lg border border-red-300 px-3 py-1.5 text-sm font-medium text-red-700 transition-colors hover:bg-red-50 disabled:cursor-not-allowed disabled:opacity-50 dark:border-red-900/60 dark:text-red-300 dark:hover:bg-red-950/40"
								>
									Reset
								</button>
							</div>
						)}
					</>
				)}
			</div>

			<ConfirmModal
				open={confirmModalOpen}
				title="Reset semantic layer"
				message="This deletes the semantic layer and a rebuild will be triggered."
				confirmLabel="Reset"
				onConfirm={handleReset}
				onCancel={() => setConfirmModalOpen(false)}
			/>

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
