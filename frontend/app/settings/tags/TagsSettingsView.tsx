// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import Link from 'next/link';
import { useRouter, useSearchParams } from 'next/navigation';
import { useCallback, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';

import { tagsApi } from '@/api/tags';
import { Button } from '@/common/Button';
import { formatDate } from '@/common/date';
import { EmptyState } from '@/common/EmptyState';
import { Icon, IconName } from '@/common/icons';
import { InfiniteScroll } from '@/common/InfiniteScroll';
import { ConfirmModal, ModalCreateNewItem } from '@/common/modal';
import { PopoverMenu } from '@/common/PopoverMenu';
import { SearchInput } from '@/common/SearchInput';
import { SkeletonRows } from '@/common/Skeleton';
import { Table } from '@/common/Table';
import { Text } from '@/common/Text';
import { AUTO_GENERATED_LABEL, MAX_TAG_NAME_LENGTH } from '@/constants/tags';
import { ButtonTheme, Size } from '@/enums/button';
import { EmptyStateVariant } from '@/enums/emptyState';
import { useDebouncedValue } from '@/hooks/useDebouncedValue';
import { useInfiniteList } from '@/hooks/useInfiniteList';
import { invalidateTagVocabulary } from '@/lib/queries/tags';
import { isProtectedTag } from '@/lib/tags';
import type { TableColumn } from '@/types/table';
import type { Tag, TagAuthor } from '@/types/tags';

import { TagDetailView } from './TagDetailView';
import { FOCUS_PARAM, TAGS_PANEL_CLASSNAME, TAGS_PANEL_PADDING, tagPath } from './tags-path';

/** As long as the Rules list waits: long enough to type a word into the box. */
const SEARCH_DEBOUNCE_MS = 1000;

/**
 * The open name dialog, and which tag it renames.
 *
 * One dialog serves both actions: the field, its length cap and the
 * duplicate-name check are the same for a new name and a changed one, and the
 * only difference is which request the submit button makes.
 */
type NameDialog = { mode: 'create' } | { mode: 'rename'; tag: Tag };

/**
 * Both timestamps are serialised from one row by one serialiser, so an
 * untouched tag has them byte-identical and no date parsing is needed to tell.
 * Rendering the creation date twice would read as an edit that never happened.
 */
const modifiedLabel = (tag: Tag): string =>
	tag.modified === tag.created ? 'Never' : formatDate(tag.modified);

/**
 * Every cell keeps its content on a 20px line — the height of an author avatar
 * — because the table aligns cells to their top edge. Without it the dates
 * would sit a few pixels above the avatar beside them.
 */
const CELL_LINE = 'text-xs leading-5 text-secondary dark:text-zinc-400';

/**
 * Who an author column names, with the initial the rest of the app draws a
 * person as.
 *
 * The resolved account, or "Auto Generated" where there is none — which covers
 * a tag the deployment generated itself, an author nobody recorded, and an id
 * whose account has since been deleted, since these ids are not foreign keys
 * and outlive the user. One answer for all three on purpose, as the note
 * beside `AUTO_GENERATED_LABEL` in `constants/tags.ts` says.
 */
const TagAuthorCell = ({ author }: { author: TagAuthor | null | undefined }) => {
	const name = author?.name || author?.email || '';
	const label = name || AUTO_GENERATED_LABEL;

	return (
		<span className="flex min-w-0 items-center gap-2">
			<span
				aria-hidden="true"
				className={`flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[10px] font-semibold ${name === '' ? 'bg-zinc-200 text-secondary dark:bg-zinc-700 dark:text-zinc-400' : 'bg-[#76b900] text-white'}`}
			>
				{label.charAt(0).toUpperCase()}
			</span>
			<span className={`min-w-0 truncate ${CELL_LINE}`}>{label}</span>
		</span>
	);
};

const TagsList = () => {
	const router = useRouter();
	// This page is the only one that writes the vocabulary, and every picker in
	// the app reads it. Each write below tells the cache so, which is what
	// keeps a tag renamed here from lingering under its old name in a picker
	// opened later in the same session.
	const queryClient = useQueryClient();
	const [dialog, setDialog] = useState<NameDialog | null>(null);
	const [name, setName] = useState('');
	const [submitting, setSubmitting] = useState(false);
	const [submitError, setSubmitError] = useState<string | null>(null);

	const [confirmDeleteTag, setConfirmDeleteTag] = useState<Tag | null>(null);
	const [deleting, setDeleting] = useState(false);
	const [deleteError, setDeleteError] = useState<string | null>(null);

	const [query, setQuery] = useState('');
	const debouncedQuery = useDebouncedValue(query.trim(), SEARCH_DEBOUNCE_MS);
	// Whether a first page has ever landed, so the box is not drawn over a list
	// that has not arrived and then withdrawn when it turns out to be empty.
	const [hasLoaded, setHasLoaded] = useState(false);

	const fetchTagsPage = useCallback(
		async (skip: number, limit: number) => {
			// The one caller that asks for authors: this is the page with the two
			// columns that show them.
			const response = await tagsApi.getAll({
				...(debouncedQuery ? { query: debouncedQuery } : {}),
				authors: true,
				skip,
				limit,
			});
			setHasLoaded(true);
			if (response.error) return { error: response.message ?? 'Failed to load tags.' };
			return { items: response.data ?? [], total: response.total ?? 0 };
		},
		[debouncedQuery],
	);

	const {
		items: tags,
		isLoading: loading,
		isLoadingMore,
		error,
		hasMore,
		loadMore,
		reload,
	} = useInfiniteList(fetchTagsPage, { itemKey: (tag) => tag.id });

	const trimmedName = name.trim();
	const renaming = dialog?.mode === 'rename' ? dialog.tag : null;
	// A rename has to change something. Submitting the identical name would
	// advance `modified` and leave the row reading as edited when it was not.
	//
	// A name another tag holds is *not* checked here, unlike a rule's: with a
	// page of the vocabulary on screen the check would read a fraction of it and
	// stay quiet about a duplicate sitting on a page nobody has scrolled to,
	// which is worse than not claiming to check at all. `uq_tag_name_lower` and
	// the DAL in front of it own the rule; the 409 they answer becomes
	// `submitError` below, in the backend's own wording.
	const unchanged = renaming != null && trimmedName === renaming.name;
	const canSubmit = !submitting && trimmedName.length > 0 && !unchanged;

	const openCreateModal = () => {
		setName('');
		setSubmitError(null);
		setDialog({ mode: 'create' });
	};

	// Opens on the tag's current name rather than empty: a rename is usually an
	// edit of what is there, and it is also what `unchanged` measures against.
	const openRenameModal = (tag: Tag) => {
		setName(tag.name);
		setSubmitError(null);
		setDialog({ mode: 'rename', tag });
	};

	const closeDialog = () => {
		if (submitting) return;
		setDialog(null);
		setSubmitError(null);
	};

	const handleSubmit = async () => {
		if (!canSubmit || dialog == null) return;
		setSubmitting(true);
		setSubmitError(null);

		const creating = dialog.mode === 'create';
		const response = creating
			? await tagsApi.create({ name: trimmedName })
			: await tagsApi.update(dialog.tag.id, { name: trimmedName });
		setSubmitting(false);

		if (response.error) {
			setSubmitError(response.message ?? `Failed to ${creating ? 'create' : 'rename'} tag.`);
			return;
		}

		setDialog(null);
		void invalidateTagVocabulary(queryClient);
		// Re-read rather than splice the answer into the rows on screen, even
		// though both requests return the whole tag: the list is ordered by name
		// and read a page at a time, so a new or renamed tag generally belongs in
		// a window nobody has loaded — and a create also moves every later tag
		// one place along, which is what the loaded pages were windows on.
		reload();
	};

	const handleRequestDelete = (tag: Tag) => {
		setDeleteError(null);
		setConfirmDeleteTag(tag);
	};

	const handleCancelDelete = () => {
		if (deleting) return;
		setConfirmDeleteTag(null);
		setDeleteError(null);
	};

	const handleConfirmDelete = async () => {
		if (confirmDeleteTag == null) return;
		setDeleting(true);
		setDeleteError(null);

		const response = await tagsApi.delete(confirmDeleteTag.id);
		setDeleting(false);

		if (response.error) {
			setDeleteError(response.message ?? 'Failed to delete tag.');
			return;
		}

		setConfirmDeleteTag(null);
		void invalidateTagVocabulary(queryClient);
		// Re-read rather than drop the row locally, for the reason the create
		// path re-reads: the pages already loaded were windows on a list one tag
		// longer, so keeping them would leave it a row short at every window
		// boundary the reader has scrolled past.
		reload();
	};

	// The list is narrowed by a query rather than by the rows on screen: with a
	// page at a time loaded, filtering what has arrived would answer from a
	// fraction of the vocabulary. Same arrangement as the Rules list.
	const searching = debouncedQuery !== '';

	const columns: TableColumn<Tag>[] = [
		{
			key: 'name',
			header: 'Name',
			// A real link rather than the row's click handler alone: it keeps
			// middle-click and "open in new tab" on the tag's name. The row stays
			// clickable too, so the whole width still opens the tag.
			cell: (tag) => (
				<Link
					href={tagPath(tag.id)}
					prefetch={false}
					onClick={(e) => e.stopPropagation()}
					className="flex min-w-0 items-center gap-3"
				>
					<Icon
						name={IconName.Tag}
						className="h-4 w-4 shrink-0 text-secondary dark:text-zinc-500"
					/>
					<Text text={tag.name} />
				</Link>
			),
			className: 'text-heading dark:text-zinc-200',
		},
		{
			key: 'created_by',
			header: 'Created By',
			width: 'w-40',
			cell: (tag) => <TagAuthorCell author={tag.created_by_user} />,
		},
		{
			key: 'created',
			header: 'Created Date',
			width: 'w-32',
			nowrap: true,
			headerClassName: 'text-right',
			className: `text-right ${CELL_LINE}`,
			cell: (tag) => formatDate(tag.created),
		},
		{
			key: 'modified_by',
			header: 'Modified By',
			width: 'w-40',
			// Blank rather than an author for a tag nothing has edited: there is no
			// editor at all, which is the same thing the date column says as "Never".
			cell: (tag) =>
				tag.modified === tag.created ? null : (
					<TagAuthorCell author={tag.modified_by_user} />
				),
		},
		{
			key: 'modified',
			header: 'Modified Date',
			width: 'w-32',
			nowrap: true,
			headerClassName: 'text-right',
			className: `text-right ${CELL_LINE}`,
			cell: (tag) => modifiedLabel(tag),
		},
		{
			key: 'actions',
			header: 'Actions',
			width: 'w-20',
			nowrap: true,
			headerClassName: 'text-right',
			cell: (tag) =>
				isProtectedTag(tag) ? null : (
					// Stop propagation so opening the menu never also opens the tag.
					<span
						role="presentation"
						onClick={(e) => e.stopPropagation()}
						className="flex justify-end"
					>
						<PopoverMenu
							items={[
								{
									label: 'Rename Tag',
									icon: <Icon name={IconName.Pencil} className="h-3.5 w-3.5" />,
									onClick: () => openRenameModal(tag),
								},
								{
									label: 'Delete Tag',
									icon: <Icon name={IconName.Trash} className="h-3.5 w-3.5" />,
									onClick: () => handleRequestDelete(tag),
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
									aria-label={`Actions for ${tag.name}`}
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
		<>
			<InfiniteScroll
				className={`flex-1 ${TAGS_PANEL_PADDING}`}
				onLoadMore={loadMore}
				isLoading={isLoadingMore}
				hasMore={hasMore}
				// Only a failed *first* page is rendered below — a failed later page
				// keeps the rows already loaded and gets its own retry control.
				error={tags.length > 0 ? error : null}
			>
				<div className="w-full space-y-5">
					<div className="flex items-center gap-3">
						<h1 className="text-base font-semibold text-heading dark:text-zinc-100">
							Tags
						</h1>
						{tags.length > 0 || searching ? (
							<div className="ml-auto">
								<Button
									theme={ButtonTheme.Primary}
									size={Size.REGULAR}
									type="button"
									onClick={openCreateModal}
									iconPosition="left"
									shadow
								>
									<Icon name={IconName.Plus} className="h-4 w-4" />
									Create New Tag
								</Button>
							</div>
						) : null}
					</div>

					{/* A box for narrowing a list, so there is none until there is a
					    list: the empty state is a page about making the first tag,
					    and a search over nothing is one more control to read past.
					    It survives a query that matches nothing, though — that list
					    is empty *because* of the box, and taking it away would leave
					    no way to clear the query and get the tags back. */}
					{hasLoaded && (tags.length > 0 || searching) ? (
						<SearchInput
							value={query}
							onChange={setQuery}
							placeholder="Search Tags…"
							aria-label="Search tags"
							className="w-full"
						/>
					) : null}

					{!loading && error != null && tags.length === 0 ? (
						<div className="rounded-lg border border-red-200/90 bg-red-50 px-4 py-3 text-sm text-red-800 dark:border-red-900/50 dark:bg-red-950/40 dark:text-red-200">
							{error}
						</div>
					) : null}

					{loading ? (
						<div className="py-2" role="status" aria-label="Loading tags">
							<SkeletonRows rows={4} />
						</div>
					) : tags.length > 0 ? (
						<Table
							columns={columns}
							rows={tags}
							rowKey={(tag) => tag.id}
							onRowClick={(tag) => router.push(tagPath(tag.id))}
							rowClickIsPointerShortcut
							containerClassName="overflow-hidden rounded-lg border border-zinc-200/90 bg-white/90 shadow-sm ring-1 ring-zinc-950/[0.04] dark:border-zinc-700/90 dark:bg-zinc-950/50 dark:ring-white/[0.06]"
							bodyClassName="divide-y divide-zinc-200/90 dark:divide-zinc-700/90"
						/>
					) : null}

					{/* A search that matches nothing is not a deployment with no
					    tags, so it says so and offers no first tag to make — the
					    box above is what to change. */}
					{!loading && error == null && tags.length === 0 ? (
						<EmptyState
							variant={EmptyStateVariant.Inline}
							icon={IconName.Tag}
							title={searching ? 'No Tags Match Your Search' : 'No Tags Yet'}
							description={
								searching
									? undefined
									: 'Tags label catalog items so they can be found and governed together.'
							}
							action={
								searching
									? undefined
									: {
											label: 'Create New Tag',
											icon: IconName.Plus,
											onClick: openCreateModal,
										}
							}
							className="rounded-lg border border-dashed border-zinc-300/90 bg-white/70 dark:border-zinc-600 dark:bg-zinc-900/30"
						/>
					) : null}
				</div>
			</InfiniteScroll>

			<ModalCreateNewItem
				open={dialog !== null}
				onClose={closeDialog}
				title={renaming == null ? 'Create New Tag' : 'Rename Tag'}
				submitLabel={renaming == null ? 'Create' : 'Save'}
				canSubmit={canSubmit}
				onSubmit={handleSubmit}
				className="w-[520px] max-w-full"
			>
				<div>
					<label
						htmlFor="tag-name"
						className="mb-1.5 flex items-center gap-2 text-sm font-semibold text-heading dark:text-zinc-100"
					>
						<Icon
							name={IconName.Tag}
							className="h-4 w-4 text-secondary dark:text-zinc-400"
						/>
						Tag Name
						<span className="font-normal text-secondary dark:text-zinc-500">
							(Max. {MAX_TAG_NAME_LENGTH})
						</span>
					</label>
					<input
						id="tag-name"
						type="text"
						value={name}
						maxLength={MAX_TAG_NAME_LENGTH}
						onChange={(e) => {
							setName(e.target.value);
							// The message names the name that was refused — "a tag
							// with this name already exists" — so editing the name
							// makes it stale rather than helpful.
							setSubmitError(null);
						}}
						placeholder="Type Tag Name"
						className={`w-full rounded-lg border bg-white px-3 py-2 text-sm text-body outline-none transition-colors placeholder:text-secondary dark:bg-zinc-900 dark:text-zinc-300 dark:placeholder:text-zinc-500 ${submitError == null ? 'border-zinc-300 focus:border-[#76b900] focus:ring-2 focus:ring-[#76b900]/30 dark:border-zinc-600' : 'border-red-400 focus:border-red-500 focus:ring-2 focus:ring-red-500/30 dark:border-red-500 dark:focus:border-red-400 dark:focus:ring-red-400/30'}`}
					/>
				</div>

				{submitError ? (
					<div className="rounded-lg border border-red-200/90 bg-red-50 px-3 py-2 text-xs text-red-800 dark:border-red-900/50 dark:bg-red-950/40 dark:text-red-200">
						{submitError}
					</div>
				) : null}

				{/* Footer rule. Negative margins cancel the modal body's padding so it
				    spans the full width, as the dialog's header rule does. */}
				<div className="-mx-6 border-t border-zinc-200 dark:border-zinc-700" />
			</ModalCreateNewItem>

			<ConfirmModal
				open={confirmDeleteTag !== null}
				title="Delete Tag"
				message={
					confirmDeleteTag == null
						? ''
						: `Are you sure you want to delete "${confirmDeleteTag.name}"? This action cannot be undone.`
				}
				onConfirm={handleConfirmDelete}
				onCancel={handleCancelDelete}
				confirming={deleting}
				error={deleteError}
			/>
		</>
	);
};

/** The panel while the boundary above resolves, so the Suspense fallback matches. */
export const TagsSettingsSkeleton = () => (
	<main className={TAGS_PANEL_CLASSNAME}>
		<div className={`w-full space-y-5 ${TAGS_PANEL_PADDING}`}>
			<h1 className="text-base font-semibold text-heading dark:text-zinc-100">Tags</h1>
			<div className="py-2" role="status" aria-label="Loading tags">
				<SkeletonRows rows={4} />
			</div>
		</div>
	</main>
);

/**
 * The Tags settings screen: the list, or one tag named by `?focus=`.
 *
 * One route rather than `/settings/tags/[tag_id]`, matching how Terms and the
 * data catalog address a selection — and `SettingsNav` keeps Tags highlighted
 * either way, since it matches on the path.
 */
export const TagsSettingsView = () => {
	const focusId = useSearchParams().get(FOCUS_PARAM);
	const focused = focusId != null && focusId !== '';

	// One frame for both views: each pages its own list and so owns the element
	// that scrolls, for the reason `TAGS_PANEL_CLASSNAME` gives.
	return (
		<main className={TAGS_PANEL_CLASSNAME}>
			{/* Keyed by the tag so switching tags remounts rather than leaving the
			    previous tag on screen while the next one loads. */}
			{focused ? <TagDetailView key={focusId} tagId={focusId} /> : <TagsList />}
		</main>
	);
};
