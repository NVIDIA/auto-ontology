// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import {
	globalSearchCountsFromResponse,
	globalSearchItemsFromResponse,
	searchApi,
} from '@/api/search';
import { Placeholders } from '@/assets/images/placeholders';
import { EmptyState } from '@/common/EmptyState';
import { searchObjectTypeFromHit } from '@/common/globalSearchMeta';
import { HeaderIcon } from '@/common/HeaderIcon';
import { Icon, IconName } from '@/common/icons';
import { GlobalSearchResults, GlobalSearchResultsSkeleton } from '@/common/GlobalSearchResults';
import {
	GlobalSearchTabs,
	GlobalSearchTabsSkeleton,
	isSearchObjectType,
	totalGlobalSearchCount,
} from '@/common/GlobalSearchTabs';
import { Modal } from '@/common/modal';
import { InputsSearch } from '@/common/InputsSearch';
import {
	GLOBAL_SEARCH_ALL_TAB,
	GLOBAL_SEARCH_LIST_LIMIT,
	GLOBAL_SEARCH_MIN_QUERY_LENGTH,
} from '@/constants/search';
import { EmptyStateVariant } from '@/enums/emptyState';
import { TextMatchOption } from '@/enums/search';
import { useDebouncedValue } from '@/hooks/useDebouncedValue';
import type { GlobalSearchItem, GlobalSearchRequest } from '@/types/search';

// Widened past names on both counts, and not yet a choice on screen: the panel
// offers no toggles, so these are what every search here runs with — and what
// every rule saved from one records.
const DEFAULT_SEARCH_FILTERS = { description: true, synonyms: true } as const;

/**
 * The search behind one tab of results.
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
	<p className="shrink-0 px-3 py-2 text-sm text-secondary dark:text-zinc-400">
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
			<GlobalSearchResults
				items={items}
				query={query}
				// Constant, because this dialog offers no filters: it always
				// searches descriptions, so a match in one is always a reason
				// the row is here.
				descriptionSearched={DEFAULT_SEARCH_FILTERS.description}
				onNavigate={onNavigate}
			/>
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
 * Split from the top bar's trigger so a page can offer its own entry point
 * without a second search state or a second copy of the trigger.
 *
 * It searches and navigates, and does nothing else with what it finds: rules
 * are built on the Discovery page, over a search whose filters are on screen
 * and part of what gets stored.
 */
export const GlobalSearchModal = ({ open, onClose }: GlobalSearchModalProps) => {
	const router = useRouter();
	const [query, setQuery] = useState('');
	const [selectedTab, setSelectedTab] = useState(GLOBAL_SEARCH_ALL_TAB);
	const [items, setItems] = useState<GlobalSearchItem[]>([]);
	const [counts, setCounts] = useState<Record<string, number>>({});
	const [resultKey, setResultKey] = useState('');
	const [countsKey, setCountsKey] = useState('');
	const [listError, setListError] = useState<string | null>(null);
	const [attempt, setAttempt] = useState(0);
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

	// Results are not cleared when the field drops below the minimum length.
	// Nothing needs hiding: `queryActive` already keeps them off screen. And
	// clearing them is what stranded the dialog on its skeleton — deleting the
	// term and typing it again inside the debounce leaves the debounced value
	// unchanged, so no request is sent, while the cleared key still reads as
	// "not answered" for ever.
	const handleQueryChange = (value: string) => {
		setQuery(value);
		setSelectedTab(GLOBAL_SEARCH_ALL_TAB);
	};

	const handleClose = () => {
		setQuery('');
		setSelectedTab(GLOBAL_SEARCH_ALL_TAB);
		resetResults();
		onClose();
	};

	/**
	 * Leave for Discovery, carrying the term across.
	 *
	 * The term as typed rather than the debounced one: the dialog is being left,
	 * so there is nothing left to wait for — a person who presses this mid-word
	 * means the word they typed, not the one the debounce still holds.
	 *
	 * Discovery re-runs it rather than only prefilling the field, and lands on
	 * the same results: its default filters search descriptions and synonyms,
	 * which is what this dialog searches. So the page opens where the dialog
	 * left off and the filters are there to narrow from, which is the whole
	 * reason to walk through this door.
	 *
	 * A term too short to search is dropped instead of sent, since Discovery
	 * would refuse to run it and the field would then hold something the
	 * results do not answer.
	 *
	 * Every press carries a fresh `n`. Without it, pressing this for the term
	 * Discovery is already showing yields the very same URL, so the page sees
	 * no navigation at all and keeps the filters and results of whatever
	 * search was narrowed in the meantime. `n` is never read as a search — it
	 * only makes each hand-over a different URL.
	 */
	const handleAdvancedSearch = () => {
		const term = query.trim();
		handleClose();
		router.push(
			term.length >= GLOBAL_SEARCH_MIN_QUERY_LENGTH
				? `/discovery?q=${encodeURIComponent(term)}&n=${Date.now()}`
				: '/discovery',
		);
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

	return (
		<Modal
			open={open}
			onClose={handleClose}
			align="top"
			overlayClassName="px-[200px] pb-4 pt-16"
			className="flex h-[min(40rem,80vh)] w-full flex-col overflow-hidden"
		>
			<div className="flex shrink-0 items-center gap-2 p-2">
				<div className="min-w-0 flex-1">
					<InputsSearch
						value={query}
						onChange={handleQueryChange}
						placeholder="Search…"
						aria-label="Search Auto Ontology"
						autoFocus
						className="w-full"
					/>
				</div>
			</div>
			<div className="flex min-h-0 flex-1 flex-col">
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
			{/* Outside the scrolling list rather than the last row of it: this is
			    the way on from a search the dialog cannot narrow, and a way on
			    that scrolls off the end of two hundred results is one nobody
			    finds. Shown whether or not anything was typed — Discovery is
			    also where a search starts from its filters rather than from a
			    term. */}
			<button
				type="button"
				onClick={handleAdvancedSearch}
				className="flex shrink-0 cursor-pointer items-center gap-2 border-t border-zinc-200 px-4 py-2.5 text-sm text-blue-600 transition-colors hover:bg-zinc-50 hover:text-blue-700 dark:border-zinc-800 dark:text-blue-400 dark:hover:bg-zinc-900/60 dark:hover:text-blue-300"
			>
				<Icon name={IconName.Filter} className="h-4 w-4 shrink-0" />
				Advanced Search
			</button>
		</Modal>
	);
};

/** The top bar's search pill and the dialog it opens. */
export const GlobalSearch = () => {
	const [open, setOpen] = useState(false);

	// ⌘/ (Ctrl+/ elsewhere) opens the search from anywhere, as the label says.
	useEffect(() => {
		const handleKeyDown = (e: KeyboardEvent) => {
			if (e.key === '/' && (e.metaKey || e.ctrlKey)) {
				e.preventDefault();
				setOpen(true);
			}
		};
		window.addEventListener('keydown', handleKeyDown);
		return () => window.removeEventListener('keydown', handleKeyDown);
	}, []);

	return (
		<>
			<button
				type="button"
				onClick={() => setOpen(true)}
				aria-label="Search Auto Ontology"
				aria-keyshortcuts="Meta+/ Control+/"
				className="group inline-flex cursor-pointer items-center rounded-[28px] bg-[#f1f8ff] py-1.5 pl-3 pr-4 text-sm leading-3 text-[#3172c6] hover:text-[#0a95ff] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-400/40 dark:bg-[rgba(155,189,190,0.12)] dark:text-[#589bf2] dark:hover:text-[#80bfff]"
			>
				<span className="flex items-center gap-1">
					<span className="group-hover:hidden">
						<HeaderIcon name="search" />
					</span>
					<span className="hidden group-hover:block">
						<HeaderIcon name="search-hover" />
					</span>
					Search Auto Ontology (⌘/)
				</span>
			</button>
			<GlobalSearchModal open={open} onClose={() => setOpen(false)} />
		</>
	);
};
