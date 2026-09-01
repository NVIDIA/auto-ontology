// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import type { Dispatch, SetStateAction } from 'react';

/**
 * Rows requested per page, unless the caller says otherwise.
 *
 * A page that fits inside the viewport leaves the sentinel visible, so the next
 * one is requested the moment it lands, and the list walks itself forward a
 * request at a time until the screen fills.
 */
export const DEFAULT_PAGE_SIZE = 10;

/** One page as the hook needs it: the rows, and how many exist in total. */
export type InfinitePage<T> = { items: T[]; total: number };
export type InfinitePageFailure = { error: string };

type FetchPage<T> = (skip: number, limit: number) => Promise<InfinitePage<T> | InfinitePageFailure>;

export type UseInfiniteListOptions<T> = {
	pageSize?: number;
	/**
	 * Identity of a row. Given one, rows already held are not added twice —
	 * worth passing when the list can be written to between page requests, since
	 * an insert shifts every later row into the next page's window.
	 */
	itemKey?: (item: T) => string;
	/** While false nothing is fetched and the list reads as empty (a closed modal, say). */
	enabled?: boolean;
};

export type UseInfiniteListResult<T> = {
	/** Every row loaded so far, in server order. */
	items: T[];
	/** Rows the query matches in full, which is what `hasMore` compares against. */
	total: number;
	/** The first page is in flight; `items` says nothing about the query yet. */
	isLoading: boolean;
	/** A further page is in flight; `items` holds what arrived before it. */
	isLoadingMore: boolean;
	error: string | null;
	hasMore: boolean;
	/** Requests the next page. A no-op while a page is in flight or at the end. */
	loadMore: () => void;
	/** Discards what is loaded and starts over from the first page. */
	reload: () => void;
	/**
	 * Rewrites the rows held, for reflecting a write the caller just made
	 * without re-reading the pages around it. `total` is left alone, so use it
	 * to patch rows rather than to add or drop them.
	 */
	setItems: Dispatch<SetStateAction<T[]>>;
};

type State<T> = {
	items: T[];
	total: number;
	/**
	 * Rows actually requested from the server so far — the sum of each raw
	 * page's size, before `itemKey` dedup. Kept apart from `items.length`
	 * because a page can arrive with fewer new rows than it held (a row
	 * already seen, e.g. after an insert/delete shifted the list between
	 * requests) without the server having re-sent anything: the *next* skip
	 * still has to be this, or it re-reads (and can re-drop) the same window
	 * forever instead of moving on.
	 */
	fetchedCount: number;
	error: string | null;
	isLoadingMore: boolean;
	/**
	 * Which query the rows above came from — the `fetchPage` identity and the
	 * reload generation. Anything else on screen means the first page of the
	 * current query hasn't arrived, which is how `isLoading` is derived instead
	 * of being flipped on from an effect.
	 */
	source: { fetchPage: unknown; reload: number } | null;
};

/** Shared so a list with nothing to show returns the same reference every render. */
const NO_ITEMS: never[] = [];

const EMPTY_STATE = {
	items: NO_ITEMS,
	total: 0,
	fetchedCount: 0,
	error: null,
	isLoadingMore: false,
	source: null,
};

const append = <T>(held: T[], incoming: T[], itemKey?: (item: T) => string): T[] => {
	if (itemKey == null) return [...held, ...incoming];
	const seen = new Set(held.map(itemKey));
	return [...held, ...incoming.filter((item) => !seen.has(itemKey(item)))];
};

/**
 * Loads a list one page at a time, keeping the pages loaded so far.
 *
 * *fetchPage* reads the window `[skip, skip + limit)` and reports the total, so
 * the hook knows when the list ends. Wrap it in `useCallback` keyed by whatever
 * the query depends on — a search string, an entity id: a new identity means a
 * different list, so the pages held stop counting and the first page is read
 * again. Late responses are discarded rather than appended to a list they don't
 * belong to.
 *
 * Pair with `InfiniteScroll`, which calls `loadMore` as the reader nears the
 * end of what is loaded.
 */
export function useInfiniteList<T>(
	fetchPage: FetchPage<T>,
	options: UseInfiniteListOptions<T> = {},
): UseInfiniteListResult<T> {
	const { pageSize = DEFAULT_PAGE_SIZE, itemKey, enabled = true } = options;
	const [state, setState] = useState<State<T>>(EMPTY_STATE);
	// Identifies the request a response belongs to. A page that arrives after
	// the query changed describes a list that is no longer on screen.
	const requestRef = useRef(0);
	// A read is outstanding. Kept alongside `isLoadingMore` because that flag
	// only reaches `loadMore` on the next render, and the scroll watcher can ask
	// twice before then — which would fetch the same page twice.
	const inFlightRef = useRef(false);
	const [reload, setReload] = useState(0);
	// Held in a ref so a caller passing `itemKey` inline — the natural way to
	// write it — doesn't give the read below a new identity on every render,
	// which would restart the effect that reads the first page without end.
	const itemKeyRef = useRef(itemKey);

	useEffect(() => {
		itemKeyRef.current = itemKey;
	}, [itemKey]);

	const load = useCallback(
		async (skip: number) => {
			requestRef.current += 1;
			const requestId = requestRef.current;
			inFlightRef.current = true;
			let page: InfinitePage<T> | InfinitePageFailure;
			try {
				page = await fetchPage(skip, pageSize);
			} catch (thrown) {
				// A `fetchPage` that throws instead of reporting `{ error }`
				// would otherwise leave the in-flight flag set for good, and
				// with it every later `loadMore`.
				page = {
					error: thrown instanceof Error ? thrown.message : 'Failed to load',
				};
			}
			// A newer read started meanwhile and owns the flag from here on.
			if (requestId !== requestRef.current) return;
			inFlightRef.current = false;

			setState((prev) => {
				const source = { fetchPage, reload };
				if ('error' in page) {
					// A failed first page has nothing to show; a failed later one
					// keeps what already loaded, so the reader isn't sent back up.
					return {
						...prev,
						source,
						items: skip === 0 ? NO_ITEMS : prev.items,
						total: skip === 0 ? 0 : prev.total,
						fetchedCount: skip === 0 ? 0 : prev.fetchedCount,
						isLoadingMore: false,
						error: page.error,
					};
				}
				return {
					source,
					items:
						skip === 0
							? page.items
							: append(prev.items, page.items, itemKeyRef.current),
					total: page.total,
					// The raw count of this page, not `items.length` above — see
					// `fetchedCount`'s doc comment on `State`.
					fetchedCount: (skip === 0 ? 0 : prev.fetchedCount) + page.items.length,
					isLoadingMore: false,
					error: null,
				};
			});
		},
		[fetchPage, pageSize, reload],
	);

	useEffect(() => {
		if (!enabled) {
			// Invalidate anything in flight, so a response for the list being
			// abandoned can't land on the next one. Nothing is cleared here: the
			// rows below read as empty until a response for this query arrives.
			requestRef.current += 1;
			return;
		}
		// Read in a closure: an effect returns its cleanup, so it can't await.
		void (async () => {
			await load(0);
		})();
	}, [enabled, load]);

	// The rows held belong to another query (or to none yet), so they say
	// nothing about this one and its first page is still on its way.
	const isStale =
		state.source == null ||
		state.source.fetchPage !== fetchPage ||
		state.source.reload !== reload;
	const showsRows = enabled && !isStale;
	const items = showsRows ? state.items : NO_ITEMS;
	const total = showsRows ? state.total : 0;
	const fetchedCount = showsRows ? state.fetchedCount : 0;

	const loadMore = useCallback(() => {
		if (inFlightRef.current || !showsRows || state.fetchedCount >= state.total) return;
		setState((prev) => ({ ...prev, isLoadingMore: true, error: null }));
		void load(state.fetchedCount);
	}, [load, showsRows, state.fetchedCount, state.total]);

	const setItems = useCallback<Dispatch<SetStateAction<T[]>>>((update) => {
		setState((prev) => ({
			...prev,
			items: typeof update === 'function' ? update(prev.items) : update,
		}));
	}, []);

	const startOver = useCallback(() => setReload((count) => count + 1), []);

	return {
		items,
		total,
		isLoading: enabled && isStale,
		isLoadingMore: state.isLoadingMore,
		error: showsRows ? state.error : null,
		// Compared against `fetchedCount`, not `items.length`: dedup can leave
		// the display short of `total` even once every row has been requested,
		// which would otherwise read as more pages existing forever.
		hasMore: fetchedCount < total,
		loadMore,
		reload: startOver,
		setItems,
	};
}
