// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useEffect, useState } from 'react';
import { rulesApi } from '@/api/rules';
import {
	globalSearchCountsFromResponse,
	globalSearchItemsFromResponse,
	searchApi,
} from '@/api/search';
import { Placeholders } from '@/assets/images/placeholders';
import { EmptyState } from '@/common/EmptyState';
import { isTaggableSearchHit, searchObjectTypeFromHit } from '@/common/globalSearchMeta';
import { GlobalSearchResults, GlobalSearchResultsSkeleton } from '@/common/GlobalSearchResults';
import {
	GlobalSearchTabs,
	GlobalSearchTabsSkeleton,
	isSearchObjectType,
	totalGlobalSearchCount,
} from '@/common/GlobalSearchTabs';
import { Modal } from '@/common/modal';
import { RuleTagPopover } from '@/common/RuleTagPopover';
import { SearchInput } from '@/common/SearchInput';
import {
	GLOBAL_SEARCH_ALL_TAB,
	GLOBAL_SEARCH_LIST_LIMIT,
	GLOBAL_SEARCH_MIN_QUERY_LENGTH,
} from '@/constants/search';
import { EmptyStateVariant } from '@/enums/emptyState';
import { TextMatchOption } from '@/enums/search';
import { useDebouncedValue } from '@/hooks/useDebouncedValue';
import { notifyRulesChanged } from '@/hooks/useRulesChanged';
import type { RuleTagDraft } from '@/types/rules';
import type { GlobalSearchItem, GlobalSearchRequest } from '@/types/search';

// Widened past names on both counts, and not yet a choice on screen: the panel
// offers no toggles, so these are what every search here runs with — and what
// every rule saved from one records.
const DEFAULT_SEARCH_FILTERS = { description: true, synonyms: true } as const;

/**
 * The search behind one tab of results.
 *
 * Shared by the list read and by the rule a person saves from it, so a rule
 * always stores the request its results actually came from — the two drifting
 * apart is exactly how a rule ends up tagging a different set than the one it
 * was created over.
 *
 * A tab other than All narrows to its own kind; All narrows to none, which is
 * `undefined` rather than every kind listed out.
 */
const globalSearchRequest = (searchTerm: string, tabId: string): GlobalSearchRequest => ({
	search_term: searchTerm,
	text_match_option: TextMatchOption.Contains,
	filters: {
		...DEFAULT_SEARCH_FILTERS,
		objects: tabId !== GLOBAL_SEARCH_ALL_TAB && isSearchObjectType(tabId) ? [tabId] : undefined,
	},
});

type GlobalSearchTabsBarProps = {
	showTabs: boolean;
	showSkeleton: boolean;
	counts: Record<string, number>;
	selected: string;
	onSelect: (tabId: string) => void;
};

const GlobalSearchTabsBar = ({
	showTabs,
	showSkeleton,
	counts,
	selected,
	onSelect,
}: GlobalSearchTabsBarProps) => {
	if (!showTabs && !showSkeleton) return null;

	return (
		<div className="shrink-0 border-b border-zinc-100 dark:border-zinc-800">
			{showTabs ? (
				<GlobalSearchTabs counts={counts} selected={selected} onSelect={onSelect} />
			) : (
				<GlobalSearchTabsSkeleton />
			)}
		</div>
	);
};

const GlobalSearchLimitBanner = () => (
	<p className="shrink-0 px-3 py-2 text-sm text-zinc-500 dark:text-zinc-400">
		Viewing top {GLOBAL_SEARCH_LIST_LIMIT} results - Try filtering to get a more accurate search
		results
	</p>
);

type GlobalSearchBodyProps = {
	showSkeleton: boolean;
	items: GlobalSearchItem[];
	query: string;
	showEmpty: boolean;
	error: string | null;
	onRetry: () => void;
	onNavigate: () => void;
};

const GlobalSearchBody = ({
	showSkeleton,
	items,
	query,
	showEmpty,
	error,
	onRetry,
	onNavigate,
}: GlobalSearchBodyProps) => (
	<div className="min-h-0 flex-1 overflow-y-auto" aria-busy={showSkeleton}>
		{showSkeleton ? <GlobalSearchResultsSkeleton /> : null}
		{items.length > 0 ? (
			<GlobalSearchResults items={items} query={query} onNavigate={onNavigate} />
		) : null}
		{error !== null ? (
			<EmptyState
				variant={EmptyStateVariant.Borderless}
				title="Search Is Unavailable"
				description={error}
				action={{ label: 'Try Again', onClick: onRetry }}
			/>
		) : null}
		{showEmpty ? (
			<EmptyState
				variant={EmptyStateVariant.Borderless}
				illustration={<Placeholders.NoResults />}
				title="No Results Match Your Search"
			/>
		) : null}
	</div>
);

export type GlobalSearchModalProps = {
	open: boolean;
	/** Called after the modal has cleared its query and results. */
	onClose: () => void;
};

/**
 * The search dialog on its own, opened by whoever owns `open`.
 *
 * Split from the top bar's trigger so a page can offer its own entry point —
 * the Rules settings screen sends people here to build a rule — without a
 * second search state or a second copy of the trigger.
 */
export const GlobalSearchModal = ({ open, onClose }: GlobalSearchModalProps) => {
	const [query, setQuery] = useState('');
	const [selectedTab, setSelectedTab] = useState(GLOBAL_SEARCH_ALL_TAB);
	const [items, setItems] = useState<GlobalSearchItem[]>([]);
	const [counts, setCounts] = useState<Record<string, number>>({});
	const [resultKey, setResultKey] = useState('');
	const [countsKey, setCountsKey] = useState('');
	const [listError, setListError] = useState<string | null>(null);
	const [attempt, setAttempt] = useState(0);
	const [ruleOpen, setRuleOpen] = useState(false);
	const debouncedQuery = useDebouncedValue(query, 1000);
	const trimmedQuery = debouncedQuery.trim();
	const liveQuery = query.trim();
	const searching = open && trimmedQuery.length >= GLOBAL_SEARCH_MIN_QUERY_LENGTH;
	const listKey = `${trimmedQuery}::${selectedTab}`;
	const loading = searching && resultKey !== listKey;
	const queryChanging =
		open && liveQuery.length >= GLOBAL_SEARCH_MIN_QUERY_LENGTH && liveQuery !== trimmedQuery;
	const awaitingSearch = queryChanging || loading;
	const countsReady = searching && countsKey === trimmedQuery;

	useEffect(() => {
		if (!open || trimmedQuery.length < GLOBAL_SEARCH_MIN_QUERY_LENGTH) return;

		const abort = new AbortController();
		void searchApi
			.globalSearch(globalSearchRequest(trimmedQuery, selectedTab), abort)
			.then((response) => {
				if (abort.signal.aborted) return;
				setListError(response.error ? (response.message ?? 'Request failed') : null);
				setItems(globalSearchItemsFromResponse(response));
				setResultKey(`${trimmedQuery}::${selectedTab}`);
			});

		return () => abort.abort();
	}, [open, trimmedQuery, selectedTab, attempt]);

	useEffect(() => {
		if (!open || trimmedQuery.length < GLOBAL_SEARCH_MIN_QUERY_LENGTH) return;

		const abort = new AbortController();
		void searchApi
			.globalSearchCount(
				{
					search_term: trimmedQuery,
					text_match_option: TextMatchOption.Contains,
					filters: DEFAULT_SEARCH_FILTERS,
				},
				abort,
			)
			.then((response) => {
				if (abort.signal.aborted || response.error) return;
				setCounts(globalSearchCountsFromResponse(response));
				setCountsKey(trimmedQuery);
			});

		return () => abort.abort();
	}, [open, trimmedQuery, attempt]);

	const resetResults = () => {
		setItems([]);
		setCounts({});
		setResultKey('');
		setCountsKey('');
		setListError(null);
	};

	// Clearing the keys puts the skeleton back while the refetch is in flight.
	const handleRetry = () => {
		resetResults();
		setAttempt((value) => value + 1);
	};

	const handleQueryChange = (value: string) => {
		setQuery(value);
		setSelectedTab(GLOBAL_SEARCH_ALL_TAB);
		if (value.trim().length < GLOBAL_SEARCH_MIN_QUERY_LENGTH) resetResults();
	};

	/**
	 * Save the rule the panel built, over the search it was built from.
	 *
	 * Owned here rather than in the panel because this is where the search lives:
	 * the panel holds the name and the tags, and the request being saved is the
	 * one these results came from.
	 *
	 * The search stays open afterwards, so the results a rule was just made over
	 * are still there to make another one from. Failures come back as a message
	 * for the panel to show rather than being handled here — the form is the only
	 * copy of what was typed, so it is the form that has to survive them.
	 *
	 * A save is announced because this dialog opens from the top bar too, over
	 * the Rules settings screen among others: the list behind it has no other
	 * way to learn that it is now a rule short of what is stored.
	 */
	const handleCreateRule = async (draft: RuleTagDraft): Promise<string | null> => {
		const response = await rulesApi.create({
			...globalSearchRequest(trimmedQuery, selectedTab),
			name: draft.name,
			tags: draft.tags,
		});
		if (response.error) return response.message ?? 'Failed to save the rule.';
		notifyRulesChanged();
		return null;
	};

	const handleClose = () => {
		setQuery('');
		setSelectedTab(GLOBAL_SEARCH_ALL_TAB);
		// The panel goes with the dialog, and it cannot report that itself: it is
		// unmounted rather than closed, so without this the next open would come
		// up with the search still dimmed.
		setRuleOpen(false);
		resetResults();
		onClose();
	};

	const queryActive = liveQuery.length >= GLOBAL_SEARCH_MIN_QUERY_LENGTH;
	const searchSettled = searching && !awaitingSearch && queryActive;
	const visibleItems = searchSettled ? items : [];
	const visibleError = searchSettled ? listError : null;
	const showEmpty = searchSettled && visibleError === null && visibleItems.length === 0;
	const showPlaceholder = !queryActive;
	const itemCounts = visibleItems.reduce<Record<string, number>>((acc, item) => {
		const kind = searchObjectTypeFromHit(item);
		acc[kind] = (acc[kind] ?? 0) + 1;
		return acc;
	}, {});
	const tabCounts = countsReady && totalGlobalSearchCount(counts) > 0 ? counts : itemCounts;
	const tabTotal =
		selectedTab === GLOBAL_SEARCH_ALL_TAB
			? totalGlobalSearchCount(tabCounts)
			: (tabCounts[selectedTab] ?? 0);
	const showTabs =
		queryActive && searching && !queryChanging && totalGlobalSearchCount(tabCounts) > 0;
	const showTabSkeleton = queryActive && awaitingSearch && !showTabs;
	const showResultSkeleton = queryActive && awaitingSearch;
	const showLimitBanner =
		visibleItems.length > 0 &&
		(visibleItems.length >= GLOBAL_SEARCH_LIST_LIMIT ||
			(countsReady && tabTotal > GLOBAL_SEARCH_LIST_LIMIT));
	// Appearance only. What actually holds these subtrees still is `inert`
	// below, which the class cannot do and must not contradict.
	const dimmedClassName = ruleOpen ? 'opacity-60' : '';
	// What the search matched, and what a rule built from it would actually
	// label. Two things separate them: the count behind the tabs is uncapped on
	// purpose — a badge has to report the real total — while the list stops at
	// `GLOBAL_SEARCH_LIST_LIMIT`, and of what the list holds only some kinds can
	// carry a tag at all (see `isTaggableSearchHit`). Offering the matched count
	// would have the panel promise 1,500 labels over a search that writes 160.
	//
	// Counted from the rows in hand rather than derived from the total, which
	// makes it exact instead of an estimate: `items` came from
	// `globalSearchRequest(trimmedQuery, selectedTab)`, the same request
	// `handleCreateRule` saves, so this *is* the list the rule replays —
	// ranking, cap and all — and these are the targets it will process.
	const matchedCount = countsReady && tabTotal > 0 ? tabTotal : visibleItems.length;
	const taggableCount = visibleItems.filter(isTaggableSearchHit).length;

	return (
		<Modal
			open={open}
			onClose={handleClose}
			align="top"
			overlayClassName="px-[200px] pb-4 pt-16"
			className="flex h-[min(40rem,80vh)] w-full flex-col overflow-hidden"
		>
			<div className="flex shrink-0 items-center gap-2 p-2">
				<div className={`min-w-0 flex-1 ${dimmedClassName}`} inert={ruleOpen}>
					<SearchInput
						value={query}
						onChange={handleQueryChange}
						placeholder="Search…"
						aria-label="Search Auto Ontology"
						autoFocus
						// `h-9` is `Size.REGULAR`'s height: the field's own padding
						// would make it 2px taller than the button standing next to it.
						className="h-9 w-full"
					/>
				</div>
				{/* A rule tags whatever the current search matches, so it can only be
				    offered once the search has matched something. */}
				{visibleItems.length > 0 ? (
					<RuleTagPopover
						itemsCount={taggableCount}
						matchedCount={matchedCount}
						onSubmit={handleCreateRule}
						onNavigate={handleClose}
						onOpenChange={setRuleOpen}
					/>
				) : null}
			</div>
			{/* Everything the rule panel is built from holds still while it is open:
			    the results are the rule's subject, so re-searching or opening one
			    from under the panel would pull the ground out from under it. An
			    outside click still closes the panel — the dismissal listens on the
			    document, not on what is under the pointer.

			    `inert` rather than `aria-hidden` beside `pointer-events-none`,
			    which is what this was: those two stop a pointer and hide the
			    subtree from a screen reader, and leave the search field, its clear
			    button and every result link in the tab order — so the one way left
			    to move the ground under the panel was the keyboard, and it moved
			    focus into content nothing was announcing. */}
			<div className={`flex min-h-0 flex-1 flex-col ${dimmedClassName}`} inert={ruleOpen}>
				<GlobalSearchTabsBar
					showTabs={showTabs}
					showSkeleton={showTabSkeleton}
					counts={tabCounts}
					selected={selectedTab}
					onSelect={setSelectedTab}
				/>
				{showLimitBanner ? <GlobalSearchLimitBanner /> : null}
				<GlobalSearchBody
					showSkeleton={showResultSkeleton}
					items={visibleItems}
					query={trimmedQuery}
					showEmpty={showEmpty || showPlaceholder}
					error={visibleError}
					onRetry={handleRetry}
					onNavigate={handleClose}
				/>
			</div>
		</Modal>
	);
};

/** The top bar's search pill and the dialog it opens. */
export const GlobalSearch = () => {
	const [open, setOpen] = useState(false);

	return (
		<>
			<button
				type="button"
				onClick={() => setOpen(true)}
				aria-label="Search Auto Ontology"
				className="inline-flex h-8 cursor-pointer items-center gap-2 rounded-full bg-blue-50 px-3 text-sm text-blue-700 transition-colors hover:bg-blue-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-400/40 dark:bg-blue-950/40 dark:text-blue-300 dark:hover:bg-blue-900/50"
			>
				<svg
					className="h-4 w-4 shrink-0"
					viewBox="0 0 20 20"
					fill="currentColor"
					aria-hidden
				>
					<path
						fillRule="evenodd"
						d="M9 3.5a5.5 5.5 0 1 0 0 11 5.5 5.5 0 0 0 0-11ZM2 9a7 7 0 1 1 12.452 4.391l3.328 3.329a.75.75 0 1 1-1.06 1.06l-3.329-3.328A7 7 0 0 1 2 9Z"
						clipRule="evenodd"
					/>
				</svg>
				Search Auto Ontology
			</button>
			<GlobalSearchModal open={open} onClose={() => setOpen(false)} />
		</>
	);
};
