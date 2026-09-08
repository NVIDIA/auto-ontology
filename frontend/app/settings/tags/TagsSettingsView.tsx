// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import Link from 'next/link';
import { useSearchParams } from 'next/navigation';
import { useEffect, useState } from 'react';

import { tagsApi } from '@/api/tags';
import { Button } from '@/common/Button';
import { formatDate } from '@/common/date';
import { EmptyState } from '@/common/EmptyState';
import { Icon, IconName } from '@/common/icons';
import { ConfirmModal, ModalCreateNewItem } from '@/common/modal';
import { PopoverMenu } from '@/common/PopoverMenu';
import { SkeletonRows } from '@/common/Skeleton';
import { SYSTEM_ACTOR, SYSTEM_ACTOR_LABEL, UNKNOWN_ACTOR_LABEL } from '@/constants/tags';
import { ButtonTheme, Size } from '@/enums/button';
import { EmptyStateVariant } from '@/enums/emptyState';
import type { Tag, TagAuthor } from '@/types/tags';

import { TagDetailView } from './TagDetailView';
import { FOCUS_PARAM, TAGS_PANEL_CLASSNAME, tagPath } from './tags-path';

/** Mirrors `MAX_TAG_NAME_LENGTH` in `gsf/server/tags/router.py`, which rejects longer. */
const MAX_TAG_NAME_LENGTH = 25;

const byName = (left: Tag, right: Tag): number =>
	left.name.toLowerCase().localeCompare(right.name.toLowerCase());

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
 * The footprint of an author column, in one place because three things have to
 * agree on it: the header, the cell, and the blank one a never-edited tag gets.
 *
 * `pl-6` is what separates an author column from the date column before it. The
 * dates are right-aligned, so their text ends at the column edge and the row's
 * `gap-3` alone leaves it almost touching the next avatar — the padding turns
 * the four columns into the two pairs they read as.
 */
const AUTHOR_COLUMN = 'w-40 shrink-0 pl-6';

/**
 * Who an author column names, with the initial the rest of the app draws a
 * person as.
 *
 * `actorId` is what the tag stored and `author` is what the API resolved it to,
 * and both are needed: a tag the deployment generated itself names no account
 * on purpose, which is a different statement from an author nobody recorded —
 * or one whose account has since been deleted, since these ids are not foreign
 * keys and outlive the user.
 */
const TagAuthorCell = ({
	actorId,
	author,
}: {
	actorId: string | null;
	author: TagAuthor | null | undefined;
}) => {
	const system = actorId === SYSTEM_ACTOR;
	const name = author?.name || author?.email || '';
	const label = system ? SYSTEM_ACTOR_LABEL : name || UNKNOWN_ACTOR_LABEL;

	return (
		<span className={`flex min-w-0 items-center gap-2 ${AUTHOR_COLUMN}`}>
			<span
				aria-hidden="true"
				className={`flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[10px] font-semibold ${name === '' ? 'bg-zinc-200 text-zinc-500 dark:bg-zinc-700 dark:text-zinc-400' : 'bg-[#76b900] text-white'}`}
			>
				{label.charAt(0).toUpperCase()}
			</span>
			<span className="min-w-0 truncate text-xs text-zinc-500 dark:text-zinc-400">
				{label}
			</span>
		</span>
	);
};

const TagsList = () => {
	const [tags, setTags] = useState<Tag[]>([]);
	const [loading, setLoading] = useState(true);
	const [error, setError] = useState<string | null>(null);

	const [dialog, setDialog] = useState<NameDialog | null>(null);
	const [name, setName] = useState('');
	const [submitting, setSubmitting] = useState(false);
	const [submitError, setSubmitError] = useState<string | null>(null);

	const [confirmDeleteTag, setConfirmDeleteTag] = useState<Tag | null>(null);
	const [deleting, setDeleting] = useState(false);
	const [deleteError, setDeleteError] = useState<string | null>(null);

	useEffect(() => {
		let cancelled = false;
		// The one caller that asks for authors: this is the page with the two
		// columns that show them.
		tagsApi.getAll({ authors: true }).then((response) => {
			if (cancelled) return;
			if (response.error) {
				setError(response.message ?? 'Failed to load tags.');
				setTags([]);
			} else {
				setError(null);
				setTags([...(response.data ?? [])].sort(byName));
			}
			setLoading(false);
		});
		return () => {
			cancelled = true;
		};
	}, []);

	const trimmedName = name.trim();
	const renaming = dialog?.mode === 'rename' ? dialog.tag : null;
	// The backend owns this rule and answers 409; checking here too is what puts
	// the message under the field while typing instead of after a round trip.
	// The tag being renamed is left out of it, because it collides with nothing
	// but itself and the backend takes its own name back — which is what makes
	// fixing a tag that was created shouting a rename rather than a duplicate.
	const nameTaken = tags.some(
		(tag) =>
			tag.id !== renaming?.id && tag.name.trim().toLowerCase() === trimmedName.toLowerCase(),
	);
	// A rename has to change something. Submitting the identical name would
	// advance `modified` and leave the row reading as edited when it was not.
	const unchanged = renaming != null && trimmedName === renaming.name;
	const canSubmit = !submitting && trimmedName.length > 0 && !nameTaken && !unchanged;

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

		// Both requests answer with the whole tag — a rename with the `modified`
		// it has just advanced — so the row is replaced by the server's version
		// rather than patched with the name that was typed.
		const saved = response.data;
		if (saved != null) {
			setTags((prev) => [...prev.filter((tag) => tag.id !== saved.id), saved].sort(byName));
		}
		setDialog(null);
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

		setTags((prev) => prev.filter((tag) => tag.id !== confirmDeleteTag.id));
		setConfirmDeleteTag(null);
	};

	return (
		<>
			<div className="w-full space-y-5">
				<div className="flex items-center gap-3">
					<h1 className="text-base font-semibold text-zinc-900 dark:text-zinc-100">
						Tags
					</h1>
					{tags.length > 0 ? (
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

				{error ? (
					<div className="rounded-lg border border-red-200/90 bg-red-50 px-4 py-3 text-sm text-red-800 dark:border-red-900/50 dark:bg-red-950/40 dark:text-red-200">
						{error}
					</div>
				) : null}

				{loading ? (
					<div className="py-2" role="status" aria-label="Loading tags">
						<SkeletonRows rows={4} />
					</div>
				) : tags.length > 0 ? (
					/* No `overflow-hidden` here, deliberately: it would clip the
					   absolutely positioned action menu of every row. */
					<div className="rounded-lg border border-zinc-200/90 bg-white/90 shadow-sm ring-1 ring-zinc-950/[0.04] dark:border-zinc-700/90 dark:bg-zinc-950/50 dark:ring-white/[0.06]">
						<div className="flex items-center gap-3 border-b border-zinc-200/90 px-4 py-2 text-xs font-semibold tracking-wide text-zinc-500 uppercase dark:border-zinc-700/90 dark:text-zinc-400">
							<span className="min-w-0 flex-1">Name</span>
							<span className={AUTHOR_COLUMN}>Created By</span>
							<span className="w-28 shrink-0 text-right">Created Date</span>
							<span className={AUTHOR_COLUMN}>Modified By</span>
							<span className="w-28 shrink-0 text-right">Modified Date</span>
							{/* The action button's exact footprint (`Size.SMALL`,
							    `iconOnly`), so the date columns line up with these
							    headers and nothing pads the right edge. */}
							<span className="w-[26px] shrink-0" aria-hidden="true" />
						</div>
						<ul className="divide-y divide-zinc-200/90 dark:divide-zinc-700/90">
							{tags.map((tag) => (
								<li
									key={tag.id}
									className="flex items-center gap-3 pr-4 transition-colors last:rounded-b-lg hover:bg-zinc-50 dark:hover:bg-zinc-800/40"
								>
									{/* A real link rather than a row-wide click handler: it
									    keeps keyboard and middle-click behaviour for free, and
									    leaving the action menu outside it is what stops the
									    menu button from also opening the tag. */}
									<Link
										href={tagPath(tag.id)}
										prefetch={false}
										className="flex min-w-0 flex-1 items-center gap-3 py-3 pl-4"
									>
										<Icon
											name={IconName.Tag}
											className="h-4 w-4 shrink-0 text-zinc-400 dark:text-zinc-500"
										/>
										<span className="min-w-0 flex-1 truncate text-sm text-zinc-800 dark:text-zinc-200">
											{tag.name}
										</span>
										<TagAuthorCell
											actorId={tag.created_by}
											author={tag.created_by_user}
										/>
										<span className="w-28 shrink-0 text-right text-xs text-zinc-500 dark:text-zinc-400">
											{formatDate(tag.created)}
										</span>
										{/* Blank rather than "Unknown" for a tag nothing has
										    edited: there is no editor to be unsure about, which
										    is the same thing the date column says as "Never". */}
										{tag.modified === tag.created ? (
											<span className={AUTHOR_COLUMN} />
										) : (
											<TagAuthorCell
												actorId={tag.modified_by}
												author={tag.modified_by_user}
											/>
										)}
										<span className="w-28 shrink-0 text-right text-xs text-zinc-500 dark:text-zinc-400">
											{modifiedLabel(tag)}
										</span>
									</Link>
									<div className="relative w-[26px] shrink-0">
										<PopoverMenu
											items={[
												{
													label: 'Rename Tag',
													icon: (
														<Icon
															name={IconName.Pencil}
															className="h-3.5 w-3.5"
														/>
													),
													onClick: () => openRenameModal(tag),
												},
												{
													label: 'Delete Tag',
													icon: (
														<Icon
															name={IconName.Trash}
															className="h-3.5 w-3.5"
														/>
													),
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
													<Icon
														name={IconName.DotsVertical}
														className="h-4 w-4"
													/>
												</Button>
											)}
										/>
									</div>
								</li>
							))}
						</ul>
					</div>
				) : (
					<EmptyState
						variant={EmptyStateVariant.Inline}
						icon={IconName.Tag}
						title="No tags yet"
						description="Tags label catalog items so they can be found and governed together."
						action={{
							label: 'Create New Tag',
							icon: IconName.Plus,
							onClick: openCreateModal,
						}}
						className="rounded-lg border border-dashed border-zinc-300/90 bg-white/70 dark:border-zinc-600 dark:bg-zinc-900/30"
					/>
				)}
			</div>

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
						className="mb-1.5 flex items-center gap-2 text-sm font-semibold text-zinc-900 dark:text-zinc-100"
					>
						<Icon
							name={IconName.Tag}
							className="h-4 w-4 text-zinc-500 dark:text-zinc-400"
						/>
						Tag Name
						<span className="font-normal text-zinc-400 dark:text-zinc-500">
							(Max. {MAX_TAG_NAME_LENGTH})
						</span>
					</label>
					<input
						id="tag-name"
						type="text"
						value={name}
						maxLength={MAX_TAG_NAME_LENGTH}
						onChange={(e) => setName(e.target.value)}
						placeholder="Type tag name"
						className={`w-full rounded-lg border bg-white px-3 py-2 text-sm text-zinc-700 outline-none transition-colors placeholder:text-zinc-400 dark:bg-zinc-900 dark:text-zinc-300 dark:placeholder:text-zinc-500 ${nameTaken ? 'border-red-400 focus:border-red-500 focus:ring-2 focus:ring-red-500/30 dark:border-red-500 dark:focus:border-red-400 dark:focus:ring-red-400/30' : 'border-zinc-300 focus:border-[#76b900] focus:ring-2 focus:ring-[#76b900]/30 dark:border-zinc-600'}`}
					/>
					{nameTaken ? (
						<p className="mt-1 text-xs text-red-500 dark:text-red-400">
							A tag with this name already exists
						</p>
					) : null}
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
				title="Delete tag"
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
		<div className="w-full space-y-5">
			<h1 className="text-base font-semibold text-zinc-900 dark:text-zinc-100">Tags</h1>
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

	return (
		<main className={TAGS_PANEL_CLASSNAME}>
			{/* Keyed by the tag so switching tags remounts rather than leaving the
			    previous tag on screen while the next one loads. */}
			{focusId != null && focusId !== '' ? (
				<TagDetailView key={focusId} tagId={focusId} />
			) : (
				<TagsList />
			)}
		</main>
	);
};
