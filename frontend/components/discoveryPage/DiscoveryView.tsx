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
import { BackPanelLayout } from '@/common/BackPanelLayout';
import { Button } from '@/common/Button';
import { EmptyState } from '@/common/EmptyState';
import { isTaggableSearchHit } from '@/common/globalSearchMeta';
import { GlobalSearchResults, GlobalSearchResultsSkeleton } from '@/common/GlobalSearchResults';
import {
	GlobalSearchTabs,
	GlobalSearchTabsSkeleton,
	isSearchObjectType,
	totalGlobalSearchCount,
} from '@/common/GlobalSearchTabs';
import { Icon, IconName } from '@/common/icons';
import { RuleTagPopover } from '@/common/RuleTagPopover';
import { SearchInput } from '@/common/SearchInput';
import { DEFAULT_DISCOVERY_FILTERS } from '@/constants/discovery';
import {
	GLOBAL_SEARCH_ALL_TAB,
	GLOBAL_SEARCH_LIST_LIMIT,
	GLOBAL_SEARCH_MIN_QUERY_LENGTH,
	UNTAGGED_TAG_FILTER,
} from '@/constants/search';
import { ButtonTheme, Size } from '@/enums/button';
import { EmptyStateVariant } from '@/enums/emptyState';
import { TextMatchOption } from '@/enums/search';
import { notifyRulesChanged } from '@/hooks/useRulesChanged';
import type { DiscoveryFilters, DiscoverySearch } from '@/types/discovery';
import type { RuleTagDraft } from '@/types/rules';
import type { GlobalSearchItem, GlobalSearchRequest } from '@/types/search';
import { DiscoveryFiltersPanel, discoveryFiltersMatchNothing } from './DiscoveryFiltersPanel';

/**
 * The request behind one tab of results, and behind a rule saved from it.
 *
 * A tab other than All narrows to its own kind instead of the filter's, which
 * is a narrowing rather than a contradiction: the tabs are drawn from a count
 * taken with the filter applied, so the only tabs on offer are kinds the
 * filter already left standing.
 */
const discoveryRequest = (search: DiscoverySearch, tabId: string): GlobalSearchRequest => {
	const tabbed = tabId !== GLOBAL_SEARCH_ALL_TAB && isSearchObjectType(tabId);
	const { objects, description, tags, data } = search.filters;

	return {
		search_term: search.term,
		text_match_option: TextMatchOption.Contains,
		filters: {
			description,
			// Sent rather than left to the backend's default, which happens to
			// agree: it is what the top bar's dialog sends, and the two have to
			// answer the same query with the same results.
			synonyms: true,
			// Null is "no narrowing" on both sides, and the request spells that
			// as an omitted field. An *empty* selection never reaches here —
			// the backend reads an empty list as every kind, the opposite of
			// what it means in the panel — so the caller answers that case
			// itself instead of asking; see `appliedMatchesNothing`.
			objects: tabbed ? [tabId] : (objects ?? undefined),
			tags: tags ?? undefined,
			data: data ?? undefined,
		},
	};
};

type DiscoveryResultsProps = {
	loading: boolean;
	searched: boolean;
	items: GlobalSearchItem[];
	query: string;
	descriptionSearched: boolean;
	error: string | null;
	onRetry: () => void;
};

const DiscoveryResults = ({
	loading,
	searched,
	items,
	query,
	descriptionSearched,
	error,
	onRetry,
}: DiscoveryResultsProps) => {
	if (loading) return <GlobalSearchResultsSkeleton />;

	if (error !== null) {
		return (
			<EmptyState
				variant={EmptyStateVariant.Borderless}
				title="Search Is Unavailable"
				description={error}
				action={{ label: 'Try Again', onClick: onRetry }}
			/>
		);
	}

	if (items.length > 0) {
		// A result here is a link to another page rather than a row inside a
		// dialog, so there is nothing for it to dismiss on the way out.
		return (
			<GlobalSearchResults
				items={items}
				query={query}
				descriptionSearched={descriptionSearched}
				onNavigate={() => {}}
			/>
		);
	}

	return (
		<EmptyState
			variant={EmptyStateVariant.Borderless}
			illustration={<Placeholders.NoResults />}
			title={searched ? 'No Results Match Your Search' : 'Search Auto Ontology'}
			description={
				searched
					? "Don't give up, search harder!"
					: 'Find terms, attributes, analyses and catalog objects by name, description or synonym.'
			}
		/>
	);
};

/**
 * The search a `?q=` arrives as, or null for a term too short to run.
 *
 * Always at the panel's defaults, which is what makes the handover honest: the
 * dialog searches with no narrowing, so the page it hands over to has to open
 * with none either — landing on a narrowed view of somebody else's search
 * would report fewer hits than the dialog just showed.
 */
const openingSearch = (term: string): DiscoverySearch | null => {
	const trimmed = term.trim();
	if (trimmed.length < GLOBAL_SEARCH_MIN_QUERY_LENGTH) return null;
	return { term: trimmed, filters: DEFAULT_DISCOVERY_FILTERS };
};

export type DiscoveryViewProps = {
	/**
	 * A search to open on, from `?q=` — the term the top bar's dialog was
	 * showing results for when Advanced Search was pressed.
	 */
	initialQuery?: string;
	/**
	 * A token that differs on every hand-over, from `?n=`. It lets the view take
	 * over a search whose term is the one already on screen, which `initialQuery`
	 * alone cannot tell apart from no navigation at all.
	 */
	handoverId?: string;
};

/**
 * Discovery: the global search as a page of its own, with the filters it
 * accepts beside it.
 *
 * Searches on Apply rather than on a debounce, which is what the filters buy:
 * a person narrowing several of them would otherwise have every intermediate
 * combination run as a query, and a rule saved from the results would be
 * saved over whichever one happened to land last.
 *
 * An `initialQuery` is the one search that runs without Apply, because it was
 * already applied elsewhere: arriving with results is the point of coming here
 * from the dialog. It is applied in the initial state rather than by an effect,
 * so there is no first paint with the term in the field and nothing beneath it.
 */
export const DiscoveryView = ({ initialQuery = '', handoverId = '' }: DiscoveryViewProps) => {
	const [query, setQuery] = useState(initialQuery);
	const [filters, setFilters] = useState<DiscoveryFilters>(DEFAULT_DISCOVERY_FILTERS);
	const [applied, setApplied] = useState<DiscoverySearch | null>(() =>
		openingSearch(initialQuery),
	);
	const [selectedTab, setSelectedTab] = useState<string>(GLOBAL_SEARCH_ALL_TAB);
	const [items, setItems] = useState<GlobalSearchItem[]>([]);
	const [counts, setCounts] = useState<Record<string, number>>({});
	const [listError, setListError] = useState<string | null>(null);
	/**
	 * Bumped by every Apply and every retry, so that re-running the search a
	 * second time over an unchanged term and unchanged filters still counts as a
	 * new search rather than as the one already answered.
	 */
	const [epoch, setEpoch] = useState(0);
	const [answeredKey, setAnsweredKey] = useState('');
	const [countedKey, setCountedKey] = useState('');
	const [countsFailed, setCountsFailed] = useState(false);
	/** Bumped by retrying the counts alone, which leaves the list as it is. */
	const [countsAttempt, setCountsAttempt] = useState(0);

	/**
	 * Whether the applied search asks for nothing, which is answered here
	 * rather than by the backend.
	 *
	 * Deselecting every object kind, or every tag, is a filter that excludes
	 * everything, and the request has no way to say so: an omitted `objects`
	 * means all of them and an empty one means the same, so asking would come
	 * back with the whole catalog — the one answer the panel is not
	 * describing. The empty result is therefore produced without a round
	 * trip, and reads on screen exactly like a search that found nothing,
	 * because that is what it is.
	 */
	const appliedMatchesNothing = applied !== null && discoveryFiltersMatchNothing(applied.filters);

	// What is in flight, and whether the list in hand is the answer to it —
	// derived rather than flagged, which keeps the whole of "loading" out of the
	// effect below and out of step with nothing. A search nobody sent is never
	// loading: no response will arrive to settle it.
	const searchKey = applied === null ? '' : `${epoch}::${selectedTab}`;
	const loading = applied !== null && !appliedMatchesNothing && answeredKey !== searchKey;
	// Tracked apart from the list, which can land first: the tab strip has to
	// stay up as a skeleton until the counts it is drawn from have answered.
	const countsKey = applied === null ? '' : `${epoch}::${countsAttempt}`;
	const countsLoading = applied !== null && !appliedMatchesNothing && countedKey !== countsKey;

	useEffect(() => {
		if (applied === null || appliedMatchesNothing) return;

		const abort = new AbortController();
		void searchApi
			.globalSearch(discoveryRequest(applied, selectedTab), abort)
			.then((response) => {
				if (abort.signal.aborted) return;
				setListError(response.error ? (response.message ?? 'Request failed') : null);
				setItems(globalSearchItemsFromResponse(response));
				setAnsweredKey(searchKey);
			});

		return () => abort.abort();
	}, [applied, appliedMatchesNothing, selectedTab, searchKey]);

	// Taken over every kind the filter left rather than over the selected tab,
	// which is what keeps the other tabs on screen once one of them is chosen.
	useEffect(() => {
		if (applied === null || appliedMatchesNothing) return;

		const abort = new AbortController();
		void searchApi
			.globalSearchCount(discoveryRequest(applied, GLOBAL_SEARCH_ALL_TAB), abort)
			.then((response) => {
				if (abort.signal.aborted) return;
				if (!response.error) setCounts(globalSearchCountsFromResponse(response));
				setCountsFailed(response.error === true);
				setCountedKey(countsKey);
			});

		return () => abort.abort();
	}, [applied, appliedMatchesNothing, countsKey]);

	const trimmedQuery = query.trim();
	// The term is the only thing Apply waits on. A filter that excludes
	// everything is a search like any other and runs like one — it just has a
	// knowable answer, which `appliedMatchesNothing` gives without asking.
	const canSearch = trimmedQuery.length >= GLOBAL_SEARCH_MIN_QUERY_LENGTH;

	const clearResults = () => {
		setItems([]);
		setCounts({});
		setListError(null);
		setAnsweredKey('');
		setCountedKey('');
		setCountsFailed(false);
		setSelectedTab(GLOBAL_SEARCH_ALL_TAB);
	};

	const runSearch = () => {
		if (!canSearch) return;
		clearResults();
		setEpoch((value) => value + 1);
		setApplied({ term: trimmedQuery, filters });
	};

	const resetAll = () => {
		setQuery('');
		setFilters(DEFAULT_DISCOVERY_FILTERS);
		clearResults();
		setApplied(null);
	};

	/**
	 * Take over a `?q=` that arrived while this page was already on screen.
	 *
	 * The dialog opens from the top bar, so Discovery is one of the pages
	 * Advanced Search is pressed *from*. That navigation keeps this component
	 * mounted — same route, same position in the tree — so the initial state
	 * above never runs again and the handed-over term would land nowhere.
	 *
	 * Adjusted during render rather than from an effect, which is what React
	 * prescribes for state derived from a changed prop: the corrected values
	 * are what reach the DOM, so there is no paint showing the previous search
	 * under the new term, and no effect that has to be kept from running twice.
	 *
	 * The filters go back to their defaults with it. They belong to the search
	 * that was on screen, not to the one arriving, and leaving them on would
	 * narrow somebody else's search without saying so.
	 *
	 * A hand-over is the term *and* its token. The term alone misses the one
	 * that matters most: Advanced Search for the term already on screen, after
	 * its filters were narrowed, changes nothing but the token — and must still
	 * start over.
	 */
	const handover = `${initialQuery}\u0000${handoverId}`;
	const [handedOver, setHandedOver] = useState(handover);
	if (handover !== handedOver) {
		setHandedOver(handover);
		setQuery(initialQuery);
		setFilters(DEFAULT_DISCOVERY_FILTERS);
		clearResults();
		setEpoch((value) => value + 1);
		setApplied(openingSearch(initialQuery));
	}

	/**
	 * Save a rule over the search these results came from.
	 *
	 * The request is rebuilt from `applied` rather than from the panel, so a
	 * rule records the search it was made from even when the filters have been
	 * changed since without being applied.
	 */
	const handleCreateRule = async (draft: RuleTagDraft): Promise<string | null> => {
		if (applied === null) return 'Run a search before saving a rule.';
		if (applied.filters.tags?.includes(UNTAGGED_TAG_FILTER) === true) {
			return 'A search for untagged objects cannot be saved as a rule.';
		}

		const response = await rulesApi.create({
			...discoveryRequest(applied, selectedTab),
			name: draft.name,
			tags: draft.tags,
		});
		if (response.error) return response.message ?? 'Failed to save the rule.';
		notifyRulesChanged();
		return null;
	};

	/**
	 * Whether the applied search asks for untagged objects, which is the one
	 * search that cannot become a rule.
	 *
	 * A rule replays its search on every pass, so one that tags what carries no
	 * tag stops matching the moment it has run: the next pass finds nothing,
	 * takes its own labels back, and the objects it tagged are untagged again.
	 * The backend refuses it with a 400; this hides the button rather than
	 * letting somebody fill a rule in and be told at the end.
	 */
	const untaggedSearch = applied?.filters.tags?.includes(UNTAGGED_TAG_FILTER) ?? false;

	const hasCounts = totalGlobalSearchCount(counts) > 0;
	const tabTotal =
		selectedTab === GLOBAL_SEARCH_ALL_TAB
			? totalGlobalSearchCount(counts)
			: (counts[selectedTab] ?? 0);
	// What the search matched against what a rule built from it could label —
	// see `RuleTagPopover`, which shows the two apart when they differ.
	const matchedCount = tabTotal > 0 ? tabTotal : items.length;
	const taggableCount = items.filter(isTaggableSearchHit).length;
	// Not while loading: after a tab change `items` is still the previous tab's
	// until the answer lands, while `tabTotal` already reads the new one, and
	// the two together would describe a list that is not on screen.
	const showLimitBanner =
		!loading &&
		items.length > 0 &&
		(items.length >= GLOBAL_SEARCH_LIST_LIMIT || tabTotal > GLOBAL_SEARCH_LIST_LIMIT);

	return (
		<div className="flex h-full w-full flex-col bg-white dark:bg-zinc-950">
			<header className="flex shrink-0 items-center gap-3 border-b border-zinc-200 px-6 py-4 dark:border-zinc-800">
				<Icon name={IconName.Discovery} className="h-5 w-5 text-body dark:text-zinc-300" />
				<h1 className="text-lg font-semibold tracking-tight text-heading dark:text-zinc-100">
					Discovery
				</h1>
			</header>
			<div className="flex min-h-0 flex-1">
				<BackPanelLayout
					panelAriaLabel="Discovery filters"
					expandAriaLabel="Expand filters"
					collapseAriaLabel="Collapse filters"
					panel={
						<DiscoveryFiltersPanel
							filters={filters}
							onChange={setFilters}
							onReset={() => setFilters(DEFAULT_DISCOVERY_FILTERS)}
						/>
					}
				>
					<main className="flex min-h-0 flex-1 flex-col">
						{/* A form so that Enter in the field applies the search, which is
						    the only reason it is one — every control inside it that is not
						    Apply is a `type="button"`. */}
						<form
							onSubmit={(event) => {
								event.preventDefault();
								runSearch();
							}}
							className="flex shrink-0 items-center gap-2 border-b border-zinc-200 px-6 py-3 dark:border-zinc-800"
						>
							<SearchInput
								value={query}
								onChange={setQuery}
								placeholder="Search…"
								aria-label="Search Auto Ontology"
								autoFocus
								// `h-9` is `Size.REGULAR`'s height: the field's own
								// padding would make it 2px taller than the buttons
								// standing next to it.
								className="h-9 min-w-0 flex-1"
							/>
							<Button type="submit" disabled={!canSearch}>
								Apply
							</Button>
							{applied !== null && (
								<Button
									theme={ButtonTheme.Outline}
									type="button"
									onClick={resetAll}
								>
									Reset
								</Button>
							)}
							{/* A rule tags whatever the current search matched, so it can
							    only be offered once the search has matched something — and
							    once the list in hand is the answer to the tab selected: during
							    a tab change `items` is still the previous tab's, so the counts
							    shown would belong to one list and the rule to another. */}
							{!loading &&
								items.length > 0 &&
								(untaggedSearch ? (
									<p className="shrink-0 text-xs text-secondary dark:text-zinc-400">
										A search for untagged objects cannot be saved as a rule
									</p>
								) : taggableCount === 0 ? (
									<p className="shrink-0 text-xs text-secondary dark:text-zinc-400">
										These results cannot carry tags, so a rule cannot be saved
										from them
									</p>
								) : (
									<RuleTagPopover
										itemsCount={taggableCount}
										matchedCount={matchedCount}
										onSubmit={handleCreateRule}
									/>
								))}
						</form>
						{applied !== null &&
							(hasCounts || loading || countsLoading || countsFailed) && (
								<div className="shrink-0 border-b border-zinc-100 px-4 pt-2 dark:border-zinc-800">
									{hasCounts ? (
										<GlobalSearchTabs
											counts={counts}
											selected={selectedTab}
											onSelect={setSelectedTab}
										/>
									) : countsFailed && !countsLoading ? (
										<div className="flex items-center gap-2 px-2 pb-2 text-sm text-secondary dark:text-zinc-400">
											<span>Result counts are unavailable.</span>
											<Button
												theme={ButtonTheme.Minimal}
												size={Size.SMALL}
												type="button"
												onClick={() => {
													setCountsFailed(false);
													setCountsAttempt((value) => value + 1);
												}}
											>
												Try Again
											</Button>
										</div>
									) : (
										<GlobalSearchTabsSkeleton />
									)}
								</div>
							)}
						{showLimitBanner && (
							<p className="shrink-0 px-6 py-2 text-sm text-secondary dark:text-zinc-400">
								Viewing the top {GLOBAL_SEARCH_LIST_LIMIT} results. Try filtering
								for more accurate results.
							</p>
						)}
						<div className="min-h-0 flex-1 overflow-y-auto px-3" aria-busy={loading}>
							<DiscoveryResults
								loading={loading}
								searched={applied !== null}
								items={items}
								query={applied?.term ?? ''}
								// Read from the applied search rather than
								// from the panel, like the query beside it:
								// changing the toggle without applying it
								// must not restyle the results already on
								// screen, which came from the old one.
								descriptionSearched={applied?.filters.description ?? false}
								error={listError}
								onRetry={() => {
									clearResults();
									setEpoch((value) => value + 1);
								}}
							/>
						</div>
					</main>
				</BackPanelLayout>
			</div>
		</div>
	);
};
