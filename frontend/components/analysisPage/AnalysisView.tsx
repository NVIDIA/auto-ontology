// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useEffect, useState } from 'react';

import { Icon, IconName } from '@/components/icons';
import { ConfirmModal, ModalCreateNewItem } from '@/components/modal';
import { SqlBlock, SqlEditor } from '@/components/SqlBlock';
import { analyses } from '@/api/analyses';
import { pqlAnalyses } from '@/api/pqlAnalyses';

export type AnalysisViewProps = Record<string, never>;

type AnalysisMode = 'sql' | 'pql';

// Normalized item so the SQL and PQL variants share one rendering/edit path.
type AnalysisItem = { id: string; name: string; description: string; code: string };

const FIELD_INPUT_CLASSNAME =
	'w-full rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm text-zinc-700 outline-none transition-colors placeholder:text-zinc-400 focus:border-[#76b900] focus:ring-2 focus:ring-[#76b900]/30 dark:border-zinc-600 dark:bg-zinc-900 dark:text-zinc-300 dark:placeholder:text-zinc-500';

const FIELD_LABEL_CLASSNAME = 'mb-1.5 block text-sm font-medium text-zinc-900 dark:text-zinc-100';

const MODE_LABEL: Record<AnalysisMode, string> = { sql: 'SQL', pql: 'PQL' };

export const AnalysisView = () => {
	const [mode, setMode] = useState<AnalysisMode>('sql');
	const [items, setItems] = useState<AnalysisItem[]>([]);
	const [loading, setLoading] = useState(true);
	const [error, setError] = useState<string | null>(null);

	const [modalOpen, setModalOpen] = useState(false);
	const [editingId, setEditingId] = useState<string | null>(null);
	const [name, setName] = useState('');
	const [description, setDescription] = useState('');
	const [code, setCode] = useState('');
	const [submitting, setSubmitting] = useState(false);
	const [submitError, setSubmitError] = useState<string | null>(null);
	const [validating, setValidating] = useState(false);
	const [validationMessage, setValidationMessage] = useState<string | null>(null);
	// SQL requires an explicit validation pass before saving; PQL does not
	// (it is validated against the prediction graph at predict time).
	const [sqlValidated, setSqlValidated] = useState(false);

	const [deletingItem, setDeletingItem] = useState<AnalysisItem | null>(null);
	const [deleting, setDeleting] = useState(false);
	const [deleteError, setDeleteError] = useState<string | null>(null);

	const isEditing = editingId !== null;
	const isPql = mode === 'pql';

	useEffect(() => {
		let cancelled = false;

		(async () => {
			setLoading(true);
			if (isPql) {
				const res = await pqlAnalyses.list();
				if (cancelled) return;
				if (res.error) {
					setError(res.message ?? 'Failed to load PQL analyses');
					setItems([]);
				} else {
					setError(null);
					setItems(
						(res.data ?? []).map((a) => ({
							id: a.id,
							name: a.name,
							description: a.description,
							code: a.pql,
						})),
					);
				}
			} else {
				const res = await analyses.list();
				if (cancelled) return;
				if (res.error) {
					setError(res.message ?? 'Failed to load custom analyses');
					setItems([]);
				} else {
					setError(null);
					setItems(
						(res.data ?? []).map((a) => ({
							id: a.id,
							name: a.name,
							description: a.description,
							code: a.sql,
						})),
					);
				}
			}
			setLoading(false);
		})();

		return () => {
			cancelled = true;
		};
	}, [mode, isPql]);

	const openCreateModal = () => {
		setEditingId(null);
		setName('');
		setDescription('');
		setCode('');
		setSubmitError(null);
		setValidationMessage(null);
		setSqlValidated(false);
		setModalOpen(true);
	};

	const openEditModal = (item: AnalysisItem) => {
		setEditingId(item.id);
		setName(item.name);
		setDescription(item.description);
		setCode(item.code);
		setSubmitError(null);
		setValidationMessage(null);
		setSqlValidated(true);
		setModalOpen(true);
	};

	const handleCodeChange = (value: string) => {
		setCode(value);
		setSqlValidated(false);
		setValidationMessage(null);
	};

	const handleClose = () => {
		if (submitting || validating) return;
		setModalOpen(false);
		setSubmitError(null);
		setValidationMessage(null);
		setSqlValidated(false);
	};

	const openDeleteModal = (item: AnalysisItem) => {
		setDeletingItem(item);
		setDeleteError(null);
	};

	const handleDeleteClose = () => {
		if (deleting) return;
		setDeletingItem(null);
		setDeleteError(null);
	};

	const handleDeleteConfirm = async () => {
		if (deletingItem == null) return;
		setDeleting(true);
		const res = isPql
			? await pqlAnalyses.delete(deletingItem.id)
			: await analyses.delete(deletingItem.id);
		setDeleting(false);
		if (res.error) {
			setDeleteError(res.message ?? `Failed to delete ${MODE_LABEL[mode]} analysis`);
			return;
		}
		setItems((prev) => prev.filter((a) => a.id !== deletingItem.id));
		setDeletingItem(null);
	};

	const trimmedName = name.trim();
	const trimmedDescription = description.trim();
	const trimmedCode = code.trim();

	const canSubmit =
		!submitting &&
		trimmedName.length > 0 &&
		trimmedDescription.length > 0 &&
		trimmedCode.length > 0 &&
		// PQL has no client-side validation gate; SQL must be validated first.
		(isPql || sqlValidated);

	const handleValidateSql = async () => {
		if (trimmedCode.length === 0) return;
		setValidating(true);
		setValidationMessage(null);
		setSqlValidated(false);
		const res = await analyses.validate(trimmedCode);
		setValidating(false);
		if (res.error) {
			setValidationMessage(res.message ?? 'SQL validation failed');
			return;
		}
		const isValid = res.data.valid === true;
		setSqlValidated(isValid);
		setValidationMessage(isValid ? 'SQL is valid.' : 'SQL validation failed');
	};

	const handleSubmit = async () => {
		if (!canSubmit) return;
		setSubmitting(true);
		setSubmitError(null);

		let savedItem: AnalysisItem | null = null;
		let failure: string | null = null;

		if (isPql) {
			const payload = {
				name: trimmedName,
				description: trimmedDescription,
				pql: trimmedCode,
			};
			const res =
				editingId !== null
					? await pqlAnalyses.update(editingId, payload)
					: await pqlAnalyses.create(payload);
			if (res.error) {
				failure = res.message ?? null;
			} else {
				savedItem = {
					id: res.data.id,
					name: res.data.name,
					description: res.data.description,
					code: res.data.pql,
				};
			}
		} else {
			const payload = {
				name: trimmedName,
				description: trimmedDescription,
				sql: trimmedCode,
			};
			const res =
				editingId !== null
					? await analyses.update(editingId, payload)
					: await analyses.create(payload);
			if (res.error) {
				failure = res.message ?? null;
			} else {
				savedItem = {
					id: res.data.id,
					name: res.data.name,
					description: res.data.description,
					code: res.data.sql,
				};
			}
		}

		setSubmitting(false);

		if (savedItem === null) {
			setSubmitError(
				failure ??
					(editingId !== null
						? `Failed to update ${MODE_LABEL[mode]} analysis`
						: `Failed to create ${MODE_LABEL[mode]} analysis`),
			);
			return;
		}

		const saved = savedItem;
		setItems((prev) => {
			if (editingId !== null) {
				return prev.map((a) => (a.id === editingId ? saved : a));
			}
			const without = prev.filter((a) => a.id !== saved.id);
			return [saved, ...without];
		});
		setModalOpen(false);
	};

	const modeButtonClass = (target: AnalysisMode) =>
		`cursor-pointer rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
			mode === target
				? 'bg-white text-zinc-900 shadow-sm dark:bg-zinc-700 dark:text-zinc-100'
				: 'text-zinc-500 hover:text-zinc-800 dark:text-zinc-400 dark:hover:text-zinc-200'
		}`;

	return (
		<div className="flex h-full w-full flex-col bg-white dark:bg-zinc-950">
			<header className="flex items-center gap-3 border-b border-zinc-200 px-6 py-4 dark:border-zinc-800">
				<Icon name={IconName.ChartBar} className="h-5 w-5 text-[#76b900]" />
				<h1 className="text-lg font-semibold tracking-tight text-zinc-900 dark:text-zinc-100">
					{isPql ? 'PQL analyses' : 'Custom analyses'}
				</h1>
				<div className="ml-4 flex items-center gap-1 rounded-lg bg-zinc-100 p-1 dark:bg-zinc-800">
					<button
						type="button"
						onClick={() => setMode('sql')}
						className={modeButtonClass('sql')}
					>
						SQL
					</button>
					<button
						type="button"
						onClick={() => setMode('pql')}
						className={modeButtonClass('pql')}
					>
						PQL
					</button>
				</div>
				<button
					type="button"
					onClick={openCreateModal}
					className="ml-auto flex cursor-pointer items-center gap-2 rounded-lg bg-[#76b900] px-4 py-2 text-sm font-medium text-white shadow-sm transition-colors hover:bg-[#5e9400]"
				>
					<svg className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor" aria-hidden>
						<path d="M10 3.75a.75.75 0 0 1 .75.75v4.75h4.75a.75.75 0 0 1 0 1.5h-4.75v4.75a.75.75 0 0 1-1.5 0V10.75H4.5a.75.75 0 0 1 0-1.5h4.75V4.5a.75.75 0 0 1 .75-.75Z" />
					</svg>
					Create new analysis
				</button>
			</header>

			<div className="flex-1 overflow-y-auto px-6 py-6">
				{loading && (
					<div className="flex h-full items-center justify-center">
						<div
							className="h-10 w-10 animate-spin rounded-full border-2 border-zinc-200 border-t-[#76b900] dark:border-zinc-700"
							role="status"
							aria-label={`Loading ${MODE_LABEL[mode]} analyses`}
						/>
					</div>
				)}

				{!loading && error != null && (
					<div className="mx-auto max-w-lg rounded-2xl border border-red-200/80 bg-white/90 px-8 py-10 text-center shadow-xl shadow-red-100/50 dark:border-red-900/50 dark:bg-zinc-950/80 dark:shadow-none">
						<h2 className="text-lg font-semibold tracking-tight text-red-800 dark:text-red-300">
							Couldn&apos;t load {MODE_LABEL[mode]} analyses
						</h2>
						<pre className="mt-4 max-w-full overflow-x-auto rounded-lg border border-red-100 bg-red-50/80 p-3 text-left text-xs text-red-900/80 dark:border-red-900/40 dark:bg-red-950/40 dark:text-red-200">
							{error}
						</pre>
					</div>
				)}

				{!loading && error == null && items.length === 0 && (
					<div className="flex h-full flex-1 items-center justify-center">
						<p className="text-sm font-medium text-zinc-700 dark:text-zinc-300">
							No {MODE_LABEL[mode]} Analyses found
						</p>
					</div>
				)}

				{!loading && error == null && items.length > 0 && (
					<ul className="flex flex-col gap-4">
						{items.map((a) => (
							<li
								key={a.id}
								className="rounded-2xl border border-zinc-200 bg-white p-5 shadow-sm dark:border-zinc-800 dark:bg-zinc-900"
							>
								<div className="flex items-start justify-between gap-3">
									<h2 className="text-base font-semibold tracking-tight text-zinc-900 dark:text-zinc-100">
										{a.name}
									</h2>
									<div className="flex shrink-0 items-center gap-1">
										<button
											type="button"
											onClick={() => openEditModal(a)}
											aria-label={`Edit ${a.name}`}
											title="Edit"
											className="cursor-pointer rounded-md p-1.5 text-zinc-500 transition-colors hover:bg-zinc-100 hover:text-[#76b900] dark:text-zinc-400 dark:hover:bg-zinc-800"
										>
											<Icon name={IconName.Pencil} className="h-4 w-4" />
										</button>
										<button
											type="button"
											onClick={() => openDeleteModal(a)}
											aria-label={`Delete ${a.name}`}
											title="Delete"
											className="cursor-pointer rounded-md p-1.5 text-zinc-500 transition-colors hover:bg-zinc-100 hover:text-red-600 dark:text-zinc-400 dark:hover:bg-zinc-800"
										>
											<Icon name={IconName.Trash} className="h-4 w-4" />
										</button>
									</div>
								</div>
								{a.description.trim() !== '' && (
									<p className="mt-2 text-sm text-zinc-600 dark:text-zinc-300">
										{a.description}
									</p>
								)}
								{a.code.trim() !== '' && (
									<SqlBlock
										sql={a.code}
										label={MODE_LABEL[mode]}
										className="mt-4"
									/>
								)}
							</li>
						))}
					</ul>
				)}
			</div>

			<ModalCreateNewItem
				open={modalOpen}
				onClose={handleClose}
				title={`${isEditing ? 'Edit' : 'Add'} ${MODE_LABEL[mode]} Analysis`}
				submitLabel={submitting ? 'Saving…' : 'Save'}
				onSubmit={handleSubmit}
				canSubmit={canSubmit}
				secondaryAction={
					isPql
						? undefined
						: {
								label: validating ? 'Validating…' : 'Validate SQL',
								onClick: handleValidateSql,
								disabled: trimmedCode.length === 0 || validating,
							}
				}
			>
				<div>
					<label className={FIELD_LABEL_CLASSNAME}>Name</label>
					<input
						type="text"
						value={name}
						onChange={(e) => setName(e.target.value)}
						placeholder={`${MODE_LABEL[mode]} Analysis Name`}
						className={FIELD_INPUT_CLASSNAME}
					/>
				</div>

				<div>
					<label className={FIELD_LABEL_CLASSNAME}>Description</label>
					<textarea
						value={description}
						onChange={(e) => setDescription(e.target.value)}
						placeholder="Add Short Description"
						rows={3}
						className={`resize-y ${FIELD_INPUT_CLASSNAME}`}
					/>
				</div>

				<div>
					<SqlEditor
						value={code}
						onChange={handleCodeChange}
						label={MODE_LABEL[mode]}
						placeholder={isPql ? 'PREDICT ...' : 'SELECT ...'}
					/>
				</div>

				{validationMessage != null && (
					<p className="rounded-lg border border-zinc-200 bg-zinc-50 px-3 py-2 text-sm text-zinc-600 dark:border-zinc-700 dark:bg-zinc-900/60 dark:text-zinc-300">
						{validationMessage}
					</p>
				)}

				{submitError != null && (
					<p className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-900/50 dark:bg-red-950/40 dark:text-red-300">
						{submitError}
					</p>
				)}
			</ModalCreateNewItem>

			<ConfirmModal
				open={deletingItem !== null}
				onCancel={handleDeleteClose}
				onConfirm={handleDeleteConfirm}
				title={`Delete ${MODE_LABEL[mode]} analysis`}
				message={
					<>
						Are you sure you want to delete <strong>{deletingItem?.name}</strong>? This
						action cannot be undone.
					</>
				}
				confirming={deleting}
				error={deleteError}
			/>
		</div>
	);
};
