// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useEffect, useState } from 'react';

import { Icon, IconName } from '@/components/icons';
import { ConfirmModal } from '@/components/ConfirmModal';
import { ModalCreateNewItem } from '@/components/ModalCreateNewItem';
import { SqlBlock, SqlEditor } from '@/components/SqlBlock';
import { analyses } from '@/api/analyses';
import type { CustomAnalysis } from '@/types/analysis';

export type AnalysisViewProps = Record<string, never>;

export const AnalysisView = () => {
	const [items, setItems] = useState<CustomAnalysis[]>([]);
	const [loading, setLoading] = useState(true);
	const [error, setError] = useState<string | null>(null);

	const [modalOpen, setModalOpen] = useState(false);
	const [editingId, setEditingId] = useState<string | null>(null);
	const [name, setName] = useState('');
	const [description, setDescription] = useState('');
	const [sql, setSql] = useState('');
	const [submitting, setSubmitting] = useState(false);
	const [submitError, setSubmitError] = useState<string | null>(null);

	const [deletingItem, setDeletingItem] = useState<CustomAnalysis | null>(null);
	const [deleting, setDeleting] = useState(false);
	const [deleteError, setDeleteError] = useState<string | null>(null);

	const isEditing = editingId !== null;

	useEffect(() => {
		let cancelled = false;

		(async () => {
			setLoading(true);
			const res = await analyses.list();
			if (cancelled) return;
			if (res.error) {
				setError(res.message ?? 'Failed to load custom analyses');
				setItems([]);
			} else {
				setError(null);
				setItems(res.data ?? []);
			}
			setLoading(false);
		})();

		return () => {
			cancelled = true;
		};
	}, []);

	const openCreateModal = () => {
		setEditingId(null);
		setName('');
		setDescription('');
		setSql('');
		setSubmitError(null);
		setModalOpen(true);
	};

	const openEditModal = (item: CustomAnalysis) => {
		setEditingId(item.id);
		setName(item.name);
		setDescription(item.description);
		setSql(item.sql);
		setSubmitError(null);
		setModalOpen(true);
	};

	const handleClose = () => {
		if (submitting) return;
		setModalOpen(false);
		setSubmitError(null);
	};

	const openDeleteModal = (item: CustomAnalysis) => {
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
		const res = await analyses.delete(deletingItem.id);
		setDeleting(false);
		if (res.error) {
			setDeleteError(res.message ?? 'Failed to delete custom analysis');
			return;
		}
		setItems((prev) => prev.filter((a) => a.id !== deletingItem.id));
		setDeletingItem(null);
	};

	const trimmedName = name.trim();
	const trimmedDescription = description.trim();
	const trimmedSql = sql.trim();

	const canSubmit =
		!submitting &&
		trimmedName.length > 0 &&
		trimmedDescription.length > 0 &&
		trimmedSql.length > 0;

	const handleSubmit = async () => {
		if (!canSubmit) return;
		setSubmitting(true);
		setSubmitError(null);

		const payload = {
			name: trimmedName,
			description: trimmedDescription,
			sql: trimmedSql,
		};

		const res =
			editingId !== null
				? await analyses.update(editingId, payload)
				: await analyses.create(payload);

		setSubmitting(false);

		if (res.error) {
			setSubmitError(
				res.message ??
					(editingId !== null
						? 'Failed to update custom analysis'
						: 'Failed to create custom analysis'),
			);
			return;
		}

		const saved = res.data;
		setItems((prev) => {
			if (editingId !== null) {
				return prev.map((a) => (a.id === editingId ? saved : a));
			}
			const without = prev.filter((a) => a.id !== saved.id);
			return [saved, ...without];
		});
		setModalOpen(false);
	};

	return (
		<div className="flex h-full w-full flex-col bg-white dark:bg-zinc-950">
			<header className="flex items-center gap-3 border-b border-zinc-200 px-6 py-4 dark:border-zinc-800">
				<Icon name={IconName.ChartBar} className="h-5 w-5 text-[#76b900]" />
				<h1 className="text-lg font-semibold tracking-tight text-zinc-900 dark:text-zinc-100">
					Custom analyses
				</h1>
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
							aria-label="Loading custom analyses"
						/>
					</div>
				)}

				{!loading && error != null && (
					<div className="mx-auto max-w-lg rounded-2xl border border-red-200/80 bg-white/90 px-8 py-10 text-center shadow-xl shadow-red-100/50 dark:border-red-900/50 dark:bg-zinc-950/80 dark:shadow-none">
						<h2 className="text-lg font-semibold tracking-tight text-red-800 dark:text-red-300">
							Couldn&apos;t load custom analyses
						</h2>
						<pre className="mt-4 max-w-full overflow-x-auto rounded-lg border border-red-100 bg-red-50/80 p-3 text-left text-xs text-red-900/80 dark:border-red-900/40 dark:bg-red-950/40 dark:text-red-200">
							{error}
						</pre>
					</div>
				)}

				{!loading && error == null && items.length === 0 && (
					<div className="flex h-full flex-1 items-center justify-center">
						<p className="text-sm font-medium text-zinc-700 dark:text-zinc-300">
							No Custom Analyses found
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
								{a.sql.trim() !== '' && <SqlBlock sql={a.sql} className="mt-4" />}
							</li>
						))}
					</ul>
				)}
			</div>

			<ModalCreateNewItem
				open={modalOpen}
				onClose={handleClose}
				title={isEditing ? 'Edit Custom Analysis' : 'Add Custom Analysis'}
				submitLabel={submitting ? 'Saving…' : 'Save'}
				onSubmit={handleSubmit}
				canSubmit={canSubmit}
			>
				<div>
					<label className="mb-1.5 block text-sm font-medium text-zinc-900 dark:text-zinc-100">
						Name
					</label>
					<input
						type="text"
						value={name}
						onChange={(e) => setName(e.target.value)}
						placeholder="Custom Analysis Name"
						className="w-full rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm text-zinc-700 outline-none transition-colors placeholder:text-zinc-400 focus:border-[#76b900] focus:ring-2 focus:ring-[#76b900]/30 dark:border-zinc-600 dark:bg-zinc-900 dark:text-zinc-300 dark:placeholder:text-zinc-500"
					/>
				</div>
				<div>
					<label className="mb-1.5 block text-sm font-medium text-zinc-900 dark:text-zinc-100">
						Description
					</label>
					<textarea
						value={description}
						onChange={(e) => setDescription(e.target.value)}
						placeholder="Add Short Description"
						rows={3}
						className="w-full resize-y rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm text-zinc-700 outline-none transition-colors placeholder:text-zinc-400 focus:border-[#76b900] focus:ring-2 focus:ring-[#76b900]/30 dark:border-zinc-600 dark:bg-zinc-900 dark:text-zinc-300 dark:placeholder:text-zinc-500"
					/>
				</div>
				<SqlEditor value={sql} onChange={setSql} />
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
				title="Delete custom analysis"
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
