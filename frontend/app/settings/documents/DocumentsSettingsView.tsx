// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useCallback, useEffect, useState } from 'react';

import { documentsApi } from '@/api/documents';
import { Button } from '@/common/Button';
import { formatDate } from '@/common/date';
import { EmptyState } from '@/common/EmptyState';
import { Icon, IconName } from '@/common/icons';
import { InfiniteScroll } from '@/common/InfiniteScroll';
import { ConfirmModal, ModalCreateNewItem } from '@/common/modal';
import { PopoverMenu } from '@/common/PopoverMenu';
import { SearchInput } from '@/common/SearchInput';
import { SkeletonRows } from '@/common/Skeleton';
import { Spinner } from '@/common/Spinner';
import { Table } from '@/common/Table';
import { Text } from '@/common/Text';
import { DOCUMENT_STATUS_POLL_MS } from '@/constants/documents';
import { ButtonTheme, Size } from '@/enums/button';
import { DocumentStatus } from '@/enums/documents';
import { EmptyStateVariant } from '@/enums/emptyState';
import { useDebouncedValue } from '@/hooks/useDebouncedValue';
import { useInfiniteList } from '@/hooks/useInfiniteList';
import type { TableColumn } from '@/types/table';
import type { DocumentItem } from '@/types/documents';

import { DocumentUploadModal } from './DocumentUploadModal';

const SEARCH_DEBOUNCE_MS = 1000;

/** The backend's page ceiling (`MAX_PAGE_SIZE` in `auto_ontology/server/pagination.py`). */
const MAX_PAGE_SIZE = 100;

const PANEL_CLASSNAME =
	'flex min-h-0 min-w-0 flex-1 flex-col bg-[linear-gradient(180deg,rgba(255,255,255,1)_0%,rgba(250,250,250,0.6)_100%)] dark:bg-[linear-gradient(180deg,rgba(9,9,11,1)_0%,rgba(24,24,27,0.5)_100%)]';
const PANEL_PADDING = 'px-7 py-6 sm:px-10 sm:py-7';

const CELL_LINE = 'text-xs leading-5 text-secondary dark:text-zinc-400';

const INPUT_CLASSNAME =
	'w-full rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm text-body outline-none transition-colors placeholder:text-secondary focus:border-[#76b900] focus:ring-2 focus:ring-[#76b900]/30 dark:border-zinc-600 dark:bg-zinc-900 dark:text-zinc-300 dark:placeholder:text-zinc-500';

/** Statuses the background task will still move on from. */
const IN_PROGRESS: ReadonlySet<DocumentStatus> = new Set([
	DocumentStatus.Processing,
	DocumentStatus.Extracting,
]);

const STATUS_BADGE: Record<DocumentStatus, { label: string; className: string }> = {
	[DocumentStatus.Processing]: {
		label: 'Processing',
		className: 'bg-amber-50 text-amber-800 dark:bg-amber-950/40 dark:text-amber-200',
	},
	[DocumentStatus.Extracting]: {
		label: 'Extracting Semantics',
		className: 'bg-sky-50 text-sky-800 dark:bg-sky-950/40 dark:text-sky-200',
	},
	[DocumentStatus.Ready]: {
		label: 'Ready',
		className: 'bg-[#76b900]/10 text-[#4a7500] dark:text-[#a3d63a]',
	},
	[DocumentStatus.Failed]: {
		label: 'Failed',
		className: 'bg-red-50 text-red-800 dark:bg-red-950/40 dark:text-red-200',
	},
};

const StatusBadge = ({ document }: { document: DocumentItem }) => {
	const badge = STATUS_BADGE[document.status];
	return (
		<span
			title={document.error ?? undefined}
			className={`inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-xs font-medium ${badge.className}`}
		>
			{IN_PROGRESS.has(document.status) ? <Spinner className="h-3 w-3" /> : null}
			{badge.label}
		</span>
	);
};

export const DocumentsSettingsView = () => {
	const [uploadOpen, setUploadOpen] = useState(false);

	const [editing, setEditing] = useState<DocumentItem | null>(null);
	const [name, setName] = useState('');
	const [description, setDescription] = useState('');
	const [submitting, setSubmitting] = useState(false);
	const [submitError, setSubmitError] = useState<string | null>(null);

	const [confirmDelete, setConfirmDelete] = useState<DocumentItem | null>(null);
	const [deleting, setDeleting] = useState(false);
	const [deleteError, setDeleteError] = useState<string | null>(null);
	const [retryingId, setRetryingId] = useState<string | null>(null);
	const [retryError, setRetryError] = useState<string | null>(null);

	const [query, setQuery] = useState('');
	const debouncedQuery = useDebouncedValue(query.trim(), SEARCH_DEBOUNCE_MS);
	const [hasLoaded, setHasLoaded] = useState(false);

	const fetchDocumentsPage = useCallback(
		async (skip: number, limit: number) => {
			const response = await documentsApi.getAll({
				...(debouncedQuery ? { query: debouncedQuery } : {}),
				skip,
				limit,
			});
			setHasLoaded(true);
			if (response.error) return { error: response.message ?? 'Failed to load documents.' };
			return { items: response.data ?? [], total: response.total ?? 0 };
		},
		[debouncedQuery],
	);

	const {
		items: documents,
		isLoading: loading,
		isLoadingMore,
		error,
		hasMore,
		loadMore,
		reload,
		setItems,
	} = useInfiniteList(fetchDocumentsPage, { itemKey: (document) => document.id });

	// Ingestion runs after the upload has answered, so a processing row only
	// learns it finished by asking again. The loaded window is re-read and its
	// rows patched in place: a `reload` would blank the list on every tick.
	const processing = documents.some((document) => IN_PROGRESS.has(document.status));
	const loadedCount = documents.length;
	useEffect(() => {
		if (!processing) return;
		const timer = setInterval(async () => {
			const response = await documentsApi.getAll({
				...(debouncedQuery ? { query: debouncedQuery } : {}),
				limit: Math.min(Math.max(loadedCount, 1), MAX_PAGE_SIZE),
			});
			if (response.error || response.data == null) return;
			const latest = new Map(response.data.map((document) => [document.id, document]));
			setItems((rows) => rows.map((row) => latest.get(row.id) ?? row));
		}, DOCUMENT_STATUS_POLL_MS);
		return () => clearInterval(timer);
	}, [processing, loadedCount, debouncedQuery, setItems]);

	const trimmedName = name.trim();
	const trimmedDescription = description.trim();
	const unchanged =
		editing != null &&
		trimmedName === editing.name &&
		trimmedDescription === (editing.description ?? '');
	const canSubmit = !submitting && trimmedName.length > 0 && !unchanged;

	const openEditModal = (document: DocumentItem) => {
		setName(document.name);
		setDescription(document.description ?? '');
		setSubmitError(null);
		setEditing(document);
	};

	const closeEditModal = () => {
		if (submitting) return;
		setEditing(null);
		setSubmitError(null);
	};

	const handleSubmit = async () => {
		if (!canSubmit || editing == null) return;
		setSubmitting(true);
		setSubmitError(null);
		const response = await documentsApi.update(editing.id, {
			name: trimmedName,
			description: trimmedDescription,
		});
		setSubmitting(false);
		if (response.error || response.data == null) {
			setSubmitError(response.message ?? 'Failed to update document.');
			return;
		}
		const updated = response.data;
		setEditing(null);
		// Patched in place: the list is ordered by date added, which an edit
		// does not change, so the row stays where it is.
		setItems((rows) => rows.map((row) => (row.id === updated.id ? updated : row)));
	};

	const handleRequestDelete = (document: DocumentItem) => {
		setDeleteError(null);
		setConfirmDelete(document);
	};

	const handleCancelDelete = () => {
		if (deleting) return;
		setConfirmDelete(null);
		setDeleteError(null);
	};

	const handleConfirmDelete = async () => {
		if (confirmDelete == null) return;
		setDeleting(true);
		setDeleteError(null);
		const response = await documentsApi.delete(confirmDelete.id);
		setDeleting(false);
		if (response.error) {
			setDeleteError(response.message ?? 'Failed to delete document.');
			return;
		}
		setConfirmDelete(null);
		reload();
	};

	const handleRetry = async (document: DocumentItem) => {
		if (retryingId != null) return;
		setRetryingId(document.id);
		setRetryError(null);
		const response = await documentsApi.retry(document.id);
		setRetryingId(null);
		if (response.error || response.data == null) {
			setRetryError(response.message ?? 'Failed to retry document.');
			return;
		}
		const updated = response.data;
		setItems((rows) => rows.map((row) => (row.id === updated.id ? updated : row)));
	};

	// The dialog closes itself once every file is sent, and stays open on the
	// ones that failed; either way the stored ones belong in the list now.
	const handleUploaded = () => reload();

	const searching = debouncedQuery !== '';

	const columns: TableColumn<DocumentItem>[] = [
		{
			key: 'name',
			header: 'Name',
			cell: (document) => <Text text={document.name} />,
			className: 'text-heading dark:text-zinc-200',
			truncate: true,
		},
		{
			key: 'description',
			header: 'Description',
			truncate: true,
			className: CELL_LINE,
			cell: (document) => <Text text={document.description || 'No Description Available'} />,
		},
		{
			key: 'file_type',
			header: 'File Type',
			width: 'w-24',
			nowrap: true,
			className: `uppercase ${CELL_LINE}`,
			cell: (document) => document.file_type,
		},
		{
			key: 'ingested_at',
			header: 'Date Added',
			width: 'w-32',
			nowrap: true,
			className: CELL_LINE,
			cell: (document) => formatDate(document.ingested_at),
		},
		{
			key: 'status',
			header: 'Status',
			width: 'w-32',
			nowrap: true,
			cell: (document) => <StatusBadge document={document} />,
		},
		{
			key: 'actions',
			header: 'Actions',
			width: 'w-20',
			nowrap: true,
			headerClassName: 'text-right',
			cell: (document) => (
				<span className="flex justify-end">
					<PopoverMenu
						items={[
							...(document.status === DocumentStatus.Failed
								? [
										{
											label:
												retryingId === document.id ? 'Retrying…' : 'Retry',
											icon: (
												<Icon
													name={IconName.Lightning}
													className="h-3.5 w-3.5"
												/>
											),
											onClick: () => void handleRetry(document),
											disabled: retryingId != null,
										},
									]
								: []),
							{
								label: 'Edit',
								icon: <Icon name={IconName.Pencil} className="h-3.5 w-3.5" />,
								onClick: () => openEditModal(document),
							},
							{
								label: 'Delete',
								icon: <Icon name={IconName.Trash} className="h-3.5 w-3.5" />,
								onClick: () => handleRequestDelete(document),
								danger: true,
							},
						]}
						trigger={({ toggle }) => (
							<Button
								theme={ButtonTheme.IconNeutral}
								size={Size.SMALL}
								iconOnly
								type="button"
								onClick={toggle}
								aria-label={`Actions for ${document.name}`}
							>
								<Icon name={IconName.DotsVertical} className="h-4 w-4" />
							</Button>
						)}
					/>
				</span>
			),
		},
	];

	return (
		<main className={PANEL_CLASSNAME}>
			<InfiniteScroll
				className={`flex-1 ${PANEL_PADDING}`}
				onLoadMore={loadMore}
				isLoading={isLoadingMore}
				hasMore={hasMore}
				error={documents.length > 0 ? error : null}
			>
				<div className="w-full space-y-5">
					<div className="flex items-center gap-3">
						<h1 className="text-base font-semibold text-heading dark:text-zinc-100">
							Documents
						</h1>
						<div className="ml-auto">
							<Button
								theme={ButtonTheme.Primary}
								size={Size.REGULAR}
								type="button"
								onClick={() => setUploadOpen(true)}
								iconPosition="left"
								shadow
							>
								<Icon name={IconName.Upload} className="h-4 w-4" />
								Upload Files
							</Button>
						</div>
					</div>

					{hasLoaded && (documents.length > 0 || searching) ? (
						<SearchInput
							value={query}
							onChange={setQuery}
							placeholder="Search Documents…"
							aria-label="Search documents"
							className="w-full"
						/>
					) : null}

					{!loading && error != null && documents.length === 0 ? (
						<div className="rounded-lg border border-red-200/90 bg-red-50 px-4 py-3 text-sm text-red-800 dark:border-red-900/50 dark:bg-red-950/40 dark:text-red-200">
							{error}
						</div>
					) : null}

					{retryError != null ? (
						<div className="rounded-lg border border-red-200/90 bg-red-50 px-4 py-3 text-sm text-red-800 dark:border-red-900/50 dark:bg-red-950/40 dark:text-red-200">
							{retryError}
						</div>
					) : null}

					{loading ? (
						<div className="py-2" role="status" aria-label="Loading documents">
							<SkeletonRows rows={4} />
						</div>
					) : documents.length > 0 ? (
						<Table
							columns={columns}
							rows={documents}
							rowKey={(document) => document.id}
							containerClassName="overflow-hidden rounded-lg border border-zinc-200/90 bg-white/90 shadow-sm ring-1 ring-zinc-950/[0.04] dark:border-zinc-700/90 dark:bg-zinc-950/50 dark:ring-white/[0.06]"
							bodyClassName="divide-y divide-zinc-200/90 dark:divide-zinc-700/90"
						/>
					) : null}

					{!loading && error == null && documents.length === 0 ? (
						<EmptyState
							variant={EmptyStateVariant.Inline}
							icon={IconName.Upload}
							title={
								searching ? 'No Documents Match Your Search' : 'No Documents Yet'
							}
							description={
								searching
									? undefined
									: 'Upload PDF, Word, PowerPoint, HTML or text files to ingest them.'
							}
							action={
								searching
									? undefined
									: {
											label: 'Upload Files',
											icon: IconName.Upload,
											onClick: () => setUploadOpen(true),
										}
							}
							className="rounded-lg border border-dashed border-zinc-300/90 bg-white/70 dark:border-zinc-600 dark:bg-zinc-900/30"
						/>
					) : null}
				</div>
			</InfiniteScroll>

			<DocumentUploadModal
				open={uploadOpen}
				onClose={() => setUploadOpen(false)}
				onUploaded={handleUploaded}
			/>

			<ModalCreateNewItem
				open={editing !== null}
				onClose={closeEditModal}
				title="Edit Document"
				submitLabel="Save"
				canSubmit={canSubmit}
				onSubmit={handleSubmit}
				className="w-[520px] max-w-full"
			>
				<div>
					<label
						htmlFor="document-name"
						className="mb-1.5 block text-sm font-semibold text-heading dark:text-zinc-100"
					>
						Name
					</label>
					<input
						id="document-name"
						type="text"
						value={name}
						onChange={(e) => {
							setName(e.target.value);
							setSubmitError(null);
						}}
						placeholder="Document name"
						className={INPUT_CLASSNAME}
					/>
				</div>
				<div>
					<label
						htmlFor="document-description"
						className="mb-1.5 block text-sm font-semibold text-heading dark:text-zinc-100"
					>
						Description
					</label>
					<textarea
						id="document-description"
						value={description}
						rows={4}
						onChange={(e) => {
							setDescription(e.target.value);
							setSubmitError(null);
						}}
						placeholder="What this document is about"
						className={`resize-none ${INPUT_CLASSNAME}`}
					/>
				</div>

				{submitError ? (
					<div className="rounded-lg border border-red-200/90 bg-red-50 px-3 py-2 text-xs text-red-800 dark:border-red-900/50 dark:bg-red-950/40 dark:text-red-200">
						{submitError}
					</div>
				) : null}
			</ModalCreateNewItem>

			<ConfirmModal
				open={confirmDelete !== null}
				title="Delete Document"
				message={
					confirmDelete == null
						? ''
						: `Are you sure you want to delete "${confirmDelete.name}"? Its extracted content is removed too. This action cannot be undone.`
				}
				onConfirm={handleConfirmDelete}
				onCancel={handleCancelDelete}
				confirming={deleting}
				error={deleteError}
			/>
		</main>
	);
};
