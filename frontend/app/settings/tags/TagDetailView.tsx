// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';

import { tagsApi } from '@/api/tags';
import { Breadcrumbs } from '@/common/Breadcrumbs';
import { formatDate } from '@/common/date';
import { EmptyState } from '@/common/EmptyState';
import { Icon, IconName } from '@/common/icons';
import { SkeletonBlock, SkeletonTable } from '@/common/Skeleton';
import { Table } from '@/common/Table';
import { EmptyStateVariant } from '@/enums/emptyState';
import { TagItemType } from '@/enums/tags';
import type { TableColumn } from '@/types/table';
import type { TagDetail, TagItem } from '@/types/tags';

import { tagItemPath } from './tag-item-path';
import { TAGS_PATH } from './tags-path';

const TYPE_LABELS: Record<TagItemType, string> = {
	[TagItemType.Term]: 'Term',
	[TagItemType.Table]: 'Table',
	[TagItemType.Column]: 'Column',
	[TagItemType.ColumnAttribute]: 'Column Attribute',
	[TagItemType.SqlAttribute]: 'SQL Attribute',
};

// A column attribute shares the Term icon because it is a property of one, and
// the Type column beside it is what names the two apart.
const TYPE_ICONS: Record<TagItemType, IconName> = {
	[TagItemType.Term]: IconName.Terms,
	[TagItemType.Table]: IconName.Table,
	[TagItemType.Column]: IconName.Column,
	[TagItemType.ColumnAttribute]: IconName.Terms,
	[TagItemType.SqlAttribute]: IconName.CodeBracket,
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
					className="h-4 w-4 shrink-0 text-zinc-400 dark:text-zinc-500"
				/>
				<span className="min-w-0 truncate text-zinc-800 dark:text-zinc-200">
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
		className: 'text-zinc-600 dark:text-zinc-400',
		cell: (item) => TYPE_LABELS[item.type],
	},
	{
		key: 'path',
		header: 'Location',
		truncate: true,
		title: (item) => item.path ?? '',
		className: 'text-zinc-600 dark:text-zinc-400',
		// Only a term has no location: it is a glossary entry, not a catalog object.
		cell: (item) => item.path ?? '—',
	},
	{
		key: 'tagged',
		header: 'Tagged',
		width: 'w-32',
		nowrap: true,
		className: 'text-zinc-500 dark:text-zinc-400',
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
 */
export const TagDetailView = ({ tagId }: TagDetailViewProps) => {
	const router = useRouter();
	const [tag, setTag] = useState<TagDetail | null>(null);
	const [loading, setLoading] = useState(true);
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
			setLoading(false);
		});
		return () => {
			cancelled = true;
		};
	}, [tagId]);

	// A row-wide handler rather than a link in the Name cell, because the whole
	// row is what a reader aims at here — and `Table` gives Enter and Space the
	// same effect, which is the part a bare `onClick` would lose. An item with
	// nowhere to open does nothing; see `tagItemPath` for when that happens.
	const handleRowClick = (item: TagItem) => {
		const path = tagItemPath(item);
		if (path != null) router.push(path);
	};

	// Nothing to head the page with, so it gets its own screen rather than an
	// error banner under a blank title. The action matters: the breadcrumb's
	// last crumb is plain text, so a lone "Tags" crumb is not a way back.
	if (error != null) {
		return (
			<div className="w-full space-y-5">
				<Breadcrumbs items={[{ label: 'Tags' }]} />
				<EmptyState
					icon={IconName.Tag}
					title="This tag could not be opened"
					description={error}
					action={{ label: 'Back to Tags', onClick: () => router.push(TAGS_PATH) }}
				/>
			</div>
		);
	}

	const items = tag?.items ?? [];

	return (
		<div className="w-full space-y-5">
			{/* The tag's own crumb is added only once its name is known: the id in
			    the URL is not a label, and an empty crumb after the separator
			    reads as a tag whose name is blank. */}
			<Breadcrumbs
				items={
					tag == null
						? [{ label: 'Tags' }]
						: [{ label: 'Tags', href: TAGS_PATH }, { label: tag.name }]
				}
			/>

			<div className="flex items-center gap-2">
				<Icon name={IconName.Tag} className="h-5 w-5 shrink-0 text-[#76b900]" />
				{tag == null ? (
					<SkeletonBlock className="h-5 w-48" />
				) : (
					<>
						<h1 className="min-w-0 truncate text-base font-semibold text-zinc-900 dark:text-zinc-100">
							{tag.name}
						</h1>
						<span className="ml-auto shrink-0 text-xs text-zinc-500 dark:text-zinc-400">
							Created {formatDate(tag.created)}
						</span>
					</>
				)}
			</div>

			{loading ? (
				<div role="status" aria-label="Loading tagged items">
					<SkeletonTable columns={4} rows={4} />
				</div>
			) : items.length > 0 ? (
				<Table
					columns={COLUMNS}
					rows={items}
					rowKey={(item) => `${item.type}:${item.id}`}
					onRowClick={handleRowClick}
					containerClassName="rounded-lg border border-zinc-200/90 bg-white/90 shadow-sm ring-1 ring-zinc-950/[0.04] dark:border-zinc-700/90 dark:bg-zinc-950/50 dark:ring-white/[0.06]"
				/>
			) : (
				<EmptyState
					variant={EmptyStateVariant.Inline}
					icon={IconName.Tag}
					title="Nothing is tagged with this tag"
					description="Terms, tables, columns, column attributes and SQL attributes carrying this tag will be listed here."
					className="rounded-lg border border-dashed border-zinc-300/90 bg-white/70 dark:border-zinc-600 dark:bg-zinc-900/30"
				/>
			)}
		</div>
	);
};
