// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { Icon, IconName } from '@/common/icons';
import {
	SEARCH_TYPE_ICON,
	SEARCH_TYPE_TAB_LABEL,
	SEARCH_TYPE_TAB_ORDER,
} from '@/common/globalSearchMeta';
import { SkeletonBlock } from '@/common/Skeleton';
import { GLOBAL_SEARCH_ALL_TAB } from '@/constants/search';
import { SearchObjectType } from '@/enums/search';
import { SkeletonVariant } from '@/enums/skeleton';

type GlobalSearchTabsProps = {
	counts: Record<string, number>;
	selected: string;
	onSelect: (tabId: string) => void;
};

const tabClass = (selected: boolean) =>
	selected
		? 'inline-flex shrink-0 cursor-pointer items-center gap-1.5 rounded-lg bg-[#76b900] px-2.5 py-1.5 text-white'
		: 'inline-flex shrink-0 cursor-pointer items-center gap-1.5 rounded-lg bg-white px-2.5 py-1.5 text-zinc-700 shadow-sm ring-1 ring-zinc-200 hover:bg-[#76b900]/10 hover:text-[#4d7a00] dark:bg-zinc-800 dark:text-zinc-200 dark:ring-zinc-700 dark:hover:bg-[#76b900]/15';

export const totalGlobalSearchCount = (counts: Record<string, number>): number =>
	Object.values(counts).reduce((sum, value) => sum + value, 0);

const TAB_SKELETON_WIDTHS = ['w-16', 'w-20', 'w-28', 'w-24', 'w-16', 'w-20'] as const;

export const GlobalSearchTabsSkeleton = () => (
	<div className="flex flex-wrap gap-2 px-2 pb-2" aria-hidden>
		{TAB_SKELETON_WIDTHS.map((width, index) => (
			<SkeletonBlock
				key={`${width}-${index}`}
				variant={SkeletonVariant.RECTANGLE}
				className={`h-8 ${width}`}
			/>
		))}
	</div>
);

export const GlobalSearchTabs = ({ counts, selected, onSelect }: GlobalSearchTabsProps) => {
	const allCount = totalGlobalSearchCount(counts);
	const typeTabs = SEARCH_TYPE_TAB_ORDER.filter((type) => (counts[type] ?? 0) > 0);

	if (allCount === 0) return null;

	return (
		<div
			role="tablist"
			aria-label="Search result types"
			className="flex flex-wrap gap-2 px-2 pb-2"
		>
			<button
				type="button"
				role="tab"
				aria-selected={selected === GLOBAL_SEARCH_ALL_TAB}
				onClick={() => onSelect(GLOBAL_SEARCH_ALL_TAB)}
				className={tabClass(selected === GLOBAL_SEARCH_ALL_TAB)}
			>
				<Icon name={IconName.Exploration} className="h-3.5 w-3.5 shrink-0" />
				<span className="text-sm font-medium">All</span>
				<span className="text-sm font-normal">{allCount.toLocaleString()}</span>
			</button>
			{typeTabs.map((type) => {
				const isSelected = selected === type;
				return (
					<button
						key={type}
						type="button"
						role="tab"
						aria-selected={isSelected}
						onClick={() => onSelect(type)}
						className={tabClass(isSelected)}
					>
						<Icon name={SEARCH_TYPE_ICON[type]} className="h-3.5 w-3.5 shrink-0" />
						<span className="text-sm font-medium">{SEARCH_TYPE_TAB_LABEL[type]}</span>
						<span className="text-sm font-normal">
							{(counts[type] ?? 0).toLocaleString()}
						</span>
					</button>
				);
			})}
		</div>
	);
};

export const isSearchObjectType = (value: string): value is SearchObjectType =>
	(Object.values(SearchObjectType) as string[]).includes(value);
