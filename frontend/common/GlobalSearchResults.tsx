// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import type { ReactNode } from 'react';
import Link from 'next/link';
import { CertificationBadge } from '@/common/CertificationBadge';
import {
	SEARCH_TYPE_ICON,
	SEARCH_TYPE_LABEL,
	searchObjectTypeFromHit,
} from '@/common/globalSearchMeta';
import { Icon, IconName } from '@/common/icons';
import { SkeletonBlock } from '@/common/Skeleton';
import { SkeletonVariant } from '@/enums/skeleton';
import { CertificationStatus } from '@/enums/certification';
import { SearchObjectType } from '@/enums/search';
import { catalogPathFromFocusId } from '@/lib/data/data-catalog-path';
import { searchTokens, synonymWordTokens } from '@/lib/searchTokens';
import type { GlobalSearchItem } from '@/types/search';

const parentTermName = (item: GlobalSearchItem): string | null => {
	const termCrumb =
		item.breadcrumbs.find((crumb) => crumb.type === SearchObjectType.Term) ??
		item.breadcrumbs[0];
	const name = termCrumb?.name?.trim();
	return name ? name : null;
};

const catalogFocusId = (item: GlobalSearchItem): string => {
	const ancestorIds = item.breadcrumbs
		.map((crumb) => crumb.id)
		.filter((id): id is string => typeof id === 'string' && id !== '');
	return [...ancestorIds, item.id].filter((id) => id !== '').join('|');
};

export const hrefForGlobalSearchItem = (item: GlobalSearchItem): string => {
	switch (searchObjectTypeFromHit(item)) {
		case SearchObjectType.Term:
			return `/terms?focus=${encodeURIComponent(item.id)}`;
		case SearchObjectType.Attribute:
			return item.parent_id
				? `/terms?focus=${encodeURIComponent(item.parent_id)}&colAttr=${encodeURIComponent(item.id)}`
				: '/terms';
		case SearchObjectType.SqlAttribute:
			return item.parent_id
				? `/terms?focus=${encodeURIComponent(item.parent_id)}&sqlAttr=${encodeURIComponent(item.id)}`
				: '/terms';
		case SearchObjectType.Analysis:
			return `/analysis?focus=${encodeURIComponent(item.id)}`;
		case SearchObjectType.PqlAnalysis:
			return `/analysis?mode=pql&focus=${encodeURIComponent(item.id)}`;
		case SearchObjectType.Db:
		case SearchObjectType.Schema:
		case SearchObjectType.Table:
		case SearchObjectType.View:
		case SearchObjectType.Column:
			return catalogPathFromFocusId(catalogFocusId(item));
		default:
			return '/data';
	}
};

const certificationStatus = (
	certified: GlobalSearchItem['certified'],
): CertificationStatus | null => {
	if (certified === true || certified === CertificationStatus.Certified) {
		return CertificationStatus.Certified;
	}
	if (certified === CertificationStatus.Partial) return CertificationStatus.Partial;
	if (certified === CertificationStatus.Pending) return CertificationStatus.Pending;
	return null;
};

const DESCRIPTION_LEAD_LENGTH = 50;

const escapeRegExp = (value: string): string => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

const tokenHighlightRegex = (tokens: string[]): RegExp => {
	const escaped = [...tokens]
		.sort((left, right) => right.length - left.length)
		.map(escapeRegExp)
		.join('|');
	return new RegExp(`(${escaped})`, 'gi');
};

const descriptionWithVisibleMatch = (description: string, query: string): string => {
	const tokens = searchTokens(query);
	if (tokens.length === 0) return description;

	const match = tokenHighlightRegex(tokens).exec(description);
	if (match == null || match.index <= DESCRIPTION_LEAD_LENGTH) return description;

	const lead = description.slice(0, DESCRIPTION_LEAD_LENGTH).trimEnd();
	return `${lead}... ${description.slice(match.index)}`;
};

const HighlightedText = ({ text, query }: { text: string; query: string }) => {
	const tokens = searchTokens(query);
	if (tokens.length === 0) return <>{text}</>;

	const needle = new Set(tokens);
	const parts = text.split(tokenHighlightRegex(tokens));

	return (
		<>
			{parts.map((part, index) =>
				needle.has(part.toLowerCase()) ? (
					<mark
						key={`${part}-${index}`}
						className="rounded-sm bg-[#76b900]/25 text-inherit"
					>
						{part}
					</mark>
				) : (
					<span key={`${part}-${index}`}>{part}</span>
				),
			)}
		</>
	);
};

const HighlightedWholeWords = ({ text, query }: { text: string; query: string }) => {
	const tokens = synonymWordTokens(query);
	if (tokens.length === 0) return <>{text}</>;
	const escaped = tokens.map(escapeRegExp).join('|');
	const matcher = new RegExp(`(^|[^a-z0-9])(${escaped})(?=[^a-z0-9]|$)`, 'gi');
	const nodes: ReactNode[] = [];
	let cursor = 0;
	for (const match of text.matchAll(matcher)) {
		const prefix = match[1] ?? '';
		const word = match[2] ?? '';
		const start = match.index ?? 0;
		const wordStart = start + prefix.length;
		if (wordStart > cursor) {
			nodes.push(text.slice(cursor, wordStart));
		}
		nodes.push(
			<mark key={`${word}-${wordStart}`} className="rounded-sm bg-[#76b900]/25 text-inherit">
				{word}
			</mark>,
		);
		cursor = wordStart + word.length;
	}
	if (cursor < text.length) nodes.push(text.slice(cursor));
	return <>{nodes}</>;
};

type GlobalSearchResultsProps = {
	items: GlobalSearchItem[];
	query: string;
	onNavigate: () => void;
};

const RESULT_SKELETON_TITLE_WIDTHS = [
	'w-2/5',
	'w-1/3',
	'w-1/2',
	'w-2/5',
	'w-1/4',
	'w-3/5',
] as const;

export const GlobalSearchResultsSkeleton = () => (
	<ul className="flex flex-col gap-2 p-3" aria-hidden>
		{RESULT_SKELETON_TITLE_WIDTHS.map((titleWidth, index) => (
			<li key={index}>
				<div className="flex min-w-0 flex-col gap-1.5 overflow-hidden rounded-xl border border-zinc-200 bg-white px-3 py-2.5 dark:border-zinc-700 dark:bg-zinc-900">
					<div className="flex min-w-0 items-center gap-2">
						<SkeletonBlock
							variant={SkeletonVariant.CIRCLE}
							className="h-4 w-4 shrink-0"
						/>
						<SkeletonBlock className={`h-4 ${titleWidth}`} />
						<SkeletonBlock
							variant={SkeletonVariant.RECTANGLE}
							className="ml-auto h-5 w-16 shrink-0 rounded-full"
						/>
					</div>
					<SkeletonBlock className={`ml-6 h-3 ${index % 2 === 0 ? 'w-1/2' : 'w-2/5'}`} />
					<SkeletonBlock className={`ml-6 h-3 ${index % 2 === 0 ? 'w-3/4' : 'w-2/3'}`} />
				</div>
			</li>
		))}
	</ul>
);

export const GlobalSearchResults = ({ items, query, onNavigate }: GlobalSearchResultsProps) => (
	<ul className="flex flex-col gap-2 p-3">
		{items.map((item) => {
			const status = certificationStatus(item.certified);
			const kind = searchObjectTypeFromHit(item);
			const isAttributeCard =
				kind === SearchObjectType.Attribute || kind === SearchObjectType.SqlAttribute;
			const termName = isAttributeCard ? parentTermName(item) : null;
			const path = isAttributeCard
				? ''
				: item.breadcrumbs.map((crumb) => crumb.name).join(' / ');
			const href = hrefForGlobalSearchItem(item);
			const typeIcon = SEARCH_TYPE_ICON[kind] ?? IconName.Table;
			const typeLabel = SEARCH_TYPE_LABEL[kind] ?? kind;

			return (
				<li key={`${item.type}:${item.id}`}>
					<Link
						href={href}
						onClick={onNavigate}
						className="flex min-w-0 cursor-pointer flex-col gap-1.5 overflow-hidden rounded-xl border border-zinc-200 bg-white px-3 py-2.5 select-none [&_*]:cursor-pointer transition-colors hover:border-[#76b900]/50 hover:bg-[#76b900]/5 dark:border-zinc-700 dark:bg-zinc-900 dark:hover:border-[#76b900]/40 dark:hover:bg-[#76b900]/10"
					>
						<div className="flex min-w-0 items-center gap-2 overflow-hidden">
							<Icon name={typeIcon} className="h-4 w-4 shrink-0 text-zinc-400" />
							<span className="min-w-0 truncate text-sm font-medium text-zinc-900 dark:text-zinc-100">
								<HighlightedText text={item.name ?? item.id} query={query} />
							</span>
							{status != null && <CertificationBadge status={status} iconOnly />}
							<span className="ml-auto shrink-0 rounded-full bg-zinc-100 px-2 py-0.5 text-[11px] font-medium text-zinc-600 dark:bg-zinc-800 dark:text-zinc-300">
								{typeLabel}
							</span>
						</div>
						{item.synonyms && item.synonyms.length > 0 ? (
							<p className="min-w-0 truncate pl-6 text-xs text-zinc-500 dark:text-zinc-400">
								<span className="text-zinc-400 dark:text-zinc-500">Synonyms: </span>
								{item.synonyms.map((synonym, index) => (
									<span key={`${item.id}-${synonym}`}>
										{index > 0 ? ', ' : null}
										<HighlightedWholeWords text={synonym} query={query} />
									</span>
								))}
							</p>
						) : null}
						{item.description ? (
							<p className="line-clamp-2 min-w-0 overflow-hidden pl-6 text-xs leading-5 text-zinc-400 dark:text-zinc-500">
								<HighlightedText
									text={descriptionWithVisibleMatch(item.description, query)}
									query={query}
								/>
							</p>
						) : null}
						{path !== '' && (
							<p className="truncate pl-6 text-xs text-zinc-500 dark:text-zinc-400">
								{path}
							</p>
						)}
						{termName != null ? (
							<p className="flex min-w-0 items-center gap-1.5 pl-6 text-xs text-zinc-500 dark:text-zinc-400">
								<Icon
									name={SEARCH_TYPE_ICON[SearchObjectType.Term]}
									className="h-3.5 w-3.5 shrink-0 text-zinc-400"
								/>
								<span className="shrink-0 text-zinc-400 dark:text-zinc-500">
									Term:{' '}
								</span>
								<span className="min-w-0 truncate">{termName}</span>
							</p>
						) : null}
					</Link>
				</li>
			);
		})}
	</ul>
);
