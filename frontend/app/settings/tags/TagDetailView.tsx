// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useRouter } from 'next/navigation';
import { useCallback, useEffect, useState } from 'react';

import { tagsApi } from '@/api/tags';
import { formatDate } from '@/common/date';
import { EmptyState } from '@/common/EmptyState';
import { Icon, IconName } from '@/common/icons';
import { InfiniteScroll } from '@/common/InfiniteScroll';
import { SkeletonBlock, SkeletonTable } from '@/common/Skeleton';
import { Table } from '@/common/Table';
import { AUTO_GENERATED_LABEL } from '@/constants/tags';
import { useBreadcrumbTrail } from '@/contexts/BreadcrumbContext';
import { EmptyStateVariant } from '@/enums/emptyState';
import { TagItemType } from '@/enums/tags';
import { useInfiniteList } from '@/hooks/useInfiniteList';
import type { TableColumn } from '@/types/table';
import type { Tag, TagItem } from '@/types/tags';

import { tagItemPath } from './tag-item-path';
import { TAGS_PANEL_PADDING, TAGS_PATH } from './tags-path';

const TYPE_LABELS: Record<TagItemType, string> = {
	[TagItemType.Term]: 'Term',
	[TagItemType.Table]: 'Table',
	[TagItemType.Column]: 'Column',
	[TagItemType.ColumnAttribute]: 'Column Attribute',
	[TagItemType.SqlAttribute]: 'SQL Attribute',
};

// The same icon each kind is drawn with in global search — see
// `SEARCH_TYPE_ICON` — since these rows lead to the same objects, and a kind
// that changed its mark between the two screens would read as another kind.
const TYPE_ICONS: Record<TagItemType, IconName> = {
	[TagItemType.Term]: IconName.Terms,
	[TagItemType.Table]: IconName.Table,
	[TagItemType.Column]: IconName.Column,
	[TagItemType.ColumnAttribute]: IconName.Key,
	[TagItemType.SqlAttribute]: IconName.CodeBracket,
};

/**
 * Who or what applied the label: the rule that matched, the person who
 * clicked, or the deployment itself.
 *
 * The `rule` first, under the same lightning mark the Rules page draws a rule
 * with — a rule labels objects nobody visited, so "which rule" is the answer
 * to why this row is here at all. Then the account, with the initial the rest
 * of the app draws a person as.
 *
 * Everything else is "Auto Generated", deliberately one answer rather than
 * two. No rule and no account means an attach that carried no identity, or a
 * row written before these columns existed, or an id whose account has since
 * been deleted — these are not foreign keys, so a label outlives its user.
 * A reader can act on "a person or a rule did this"; they can do nothing with
 * the difference between an id that resolves to nobody and no id at all, so
 * this column does not spend a word on it — nor does any other column in the
 * app that names a person.
 */
const TaggedByCell = ({ item }: { item: TagItem }) => {
	if (item.rule != null) {
		return (
			<span className="flex min-w-0 items-center gap-1.5" title={item.rule.name}>
				<Icon
					name={IconName.Lightning}
					className="h-3.5 w-3.5 shrink-0 text-body dark:text-zinc-300"
				/>
				<span className="min-w-0 truncate text-body dark:text-zinc-300">
					{item.rule.name}
				</span>
			</span>
		);
	}

	const name = item.tagged_by_user?.name || item.tagged_by_user?.email || '';
	const known = name !== '';
	const label = known ? name : AUTO_GENERATED_LABEL;

	return (
		<span className="flex min-w-0 items-center gap-2" title={label}>
			<span
				aria-hidden="true"
				className={`flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[10px] font-semibold ${known ? 'bg-[#76b900] text-white' : 'bg-zinc-200 text-secondary dark:bg-zinc-700 dark:text-zinc-400'}`}
			>
				{label.charAt(0).toUpperCase()}
			</span>
			<span className="min-w-0 truncate text-body dark:text-zinc-300">{label}</span>
		</span>
	);
};

const COLUMNS: TableColumn<TagItem>[] = [
	{
		key: 'name',
		header: 'Name',
		truncate: true,
		title: (item) => item.name,
		cell: (item) => (
			<span className="flex min-w-0 items-center gap-2">
				<Icon
					name={TYPE_ICONS[item.type]}
					className="h-4 w-4 shrink-0 text-secondary dark:text-zinc-500"
				/>
				<span className="min-w-0 truncate text-heading dark:text-zinc-200">
					{item.name}
				</span>
			</span>
		),
	},
	{
		key: 'type',
		header: 'Type',
		width: 'w-44',
		nowrap: true,
		className: 'text-body dark:text-zinc-400',
		cell: (item) => TYPE_LABELS[item.type],
	},
	{
		key: 'path',
		header: 'Location',
		truncate: true,
		title: (item) => item.path ?? '',
		className: 'text-body dark:text-zinc-400',
		// Only a term has no location: it is a glossary entry, not a catalog object.
		cell: (item) => item.path ?? '—',
	},
	{
		key: 'tagged_by',
		header: 'Tagged By',
		width: 'w-48',
		nowrap: true,
		cell: (item) => <TaggedByCell item={item} />,
	},
	{
		key: 'tagged',
		header: 'Tagged',
		width: 'w-32',
		nowrap: true,
		className: 'text-secondary dark:text-zinc-400',
		cell: (item) => formatDate(item.tagged),
	},
];

type TagDetailViewProps = {
	tagId: string;
};

/**
 * One tag and everything it labels.
 *
 * Read-only: tags are applied from the object's own page — the Tags section on
 * a term, say — so this is where you see what a tag has come to mean across the
 * deployment, not where you edit it. Which is also why every row opens that
 * page: it is where the labelling can actually be changed. A tag nothing
 * carries gets the empty state below rather than a blank table.
 *
 * Two reads rather than one, and the second is paged: a tag applied by a rule
 * lands on everything a search matched, and goes on landing on whatever matches
 * later, so what a tag labels is a list without a known ceiling. The rows are
 * read a page at a time as the reader scrolls, while the tag itself — the name
 * this page is titled with — is one small read that does not wait for them.
 */
export const TagDetailView = ({ tagId }: TagDetailViewProps) => {
	const router = useRouter();
	// No loading flag beside these: the two reads render independently, and a
	// tag that has not arrived is `tag == null`, which is what the title
	// skeletons on. One flag covering both would tie the table to the title.
	const [tag, setTag] = useState<Tag | null>(null);
	const [error, setError] = useState<string | null>(null);

	useEffect(() => {
		let cancelled = false;
		tagsApi.getById(tagId).then((response) => {
			if (cancelled) return;
			if (response.error) {
				setError(response.message ?? 'Failed to load tag.');
				setTag(null);
			} else {
				setError(null);
				setTag(response.data ?? null);
			}
		});
		return () => {
			cancelled = true;
		};
	}, [tagId]);

	const fetchItemsPage = useCallback(
		async (skip: number, limit: number) => {
			const response = await tagsApi.getTargets(tagId, { skip, limit });
			if (response.error) {
				return { error: response.message ?? 'Failed to load tagged items.' };
			}
			return { items: response.data ?? [], total: response.total ?? 0 };
		},
		[tagId],
	);

	const {
		items,
		total,
		isLoading: itemsLoading,
		isLoadingMore,
		error: itemsError,
		hasMore,
		loadMore,
	} = useInfiniteList(fetchItemsPage, {
		// An object can only carry a tag once, so a row cannot legitimately
		// repeat — but a label applied while the reader is scrolling shifts every
		// later row into the next window, which is exactly when a page re-sends
		// one it already sent.
		itemKey: (item) => `${item.type}:${item.id}`,
	});

	// A row-wide handler rather than a link in the Name cell, because the whole
	// row is what a reader aims at here — and `Table` gives Enter and Space the
	// same effect, which is the part a bare `onClick` would lose. An item with
	// nowhere to open does nothing; see `tagItemPath` for when that happens.
	const handleRowClick = (item: TagItem) => {
		const path = tagItemPath(item);
		if (path != null) router.push(path);
	};

	useBreadcrumbTrail(tag == null ? [] : [{ label: tag.name }]);

	// Nothing to head the page with, so it gets its own screen rather than an
	// error banner under a blank title. The action matters: the trail ends at
	// "Tags" here, and a crumb the reader is already on is not a way back.
	if (error != null) {
		return (
			<div className={`w-full space-y-5 ${TAGS_PANEL_PADDING}`}>
				<EmptyState
					icon={IconName.Tag}
					title="This tag could not be opened"
					description={error}
					action={{ label: 'Back to Tags', onClick: () => router.push(TAGS_PATH) }}
				/>
			</div>
		);
	}

	return (
		<InfiniteScroll
			className={`flex-1 ${TAGS_PANEL_PADDING}`}
			onLoadMore={loadMore}
			isLoading={isLoadingMore}
			hasMore={hasMore}
			// Only a failed *first* page is rendered below — a failed later page
			// keeps the rows already loaded and gets its own retry control.
			error={items.length > 0 ? itemsError : null}
		>
			<div className="w-full space-y-5">
				<div className="flex items-center gap-2">
					<Icon
						name={IconName.Tag}
						className="h-5 w-5 shrink-0 text-body dark:text-zinc-300"
					/>
					{tag == null ? (
						<SkeletonBlock className="h-5 w-48" />
					) : (
						<>
							<h1 className="min-w-0 truncate text-base font-semibold text-heading dark:text-zinc-100">
								{tag.name}
							</h1>
							{/* How many objects carry the tag, which a scrolled list
							    cannot say by its length: what is on screen is as far
							    as the reader has got, not how far there is to go.
							    Withheld until a page has landed, since a list that
							    failed to load has no count rather than a count of
							    none. */}
							<span className="ml-auto shrink-0 text-xs text-secondary dark:text-zinc-400">
								{itemsLoading || itemsError != null ? null : `${total} tagged · `}
								Created {formatDate(tag.created)}
							</span>
						</>
					)}
				</div>

				{!itemsLoading && itemsError != null && items.length === 0 ? (
					<div className="rounded-lg border border-red-200/90 bg-red-50 px-4 py-3 text-sm text-red-800 dark:border-red-900/50 dark:bg-red-950/40 dark:text-red-200">
						{itemsError}
					</div>
				) : null}

				{itemsLoading ? (
					<div role="status" aria-label="Loading tagged items">
						<SkeletonTable columns={5} rows={4} />
					</div>
				) : null}

				{items.length > 0 ? (
					<Table
						columns={COLUMNS}
						rows={items}
						rowKey={(item) => `${item.type}:${item.id}`}
						onRowClick={handleRowClick}
						containerClassName="rounded-lg border border-zinc-200/90 bg-white/90 shadow-sm ring-1 ring-zinc-950/[0.04] dark:border-zinc-700/90 dark:bg-zinc-950/50 dark:ring-white/[0.06]"
					/>
				) : null}

				{!itemsLoading && itemsError == null && items.length === 0 ? (
					<EmptyState
						variant={EmptyStateVariant.Inline}
						icon={IconName.Tag}
						title="Nothing is tagged with this tag"
						description="Terms, tables, columns, column attributes and SQL attributes carrying this tag will be listed here."
						className="rounded-lg border border-dashed border-zinc-300/90 bg-white/70 dark:border-zinc-600 dark:bg-zinc-900/30"
					/>
				) : null}
			</div>
		</InfiniteScroll>
	);
};
