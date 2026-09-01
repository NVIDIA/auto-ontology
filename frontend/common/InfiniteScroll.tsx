// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useEffect, useRef } from 'react';
import type { ReactNode } from 'react';
import { Spinner } from '@nvidia/foundations-react-core';

import { Button } from '@/common/Button';
import { ButtonTheme, Size } from '@/enums/button';

type InfiniteScrollProps = {
	/**
	 * Asks for the next page. Called when the end of the list comes into view,
	 * and again after each page lands while the end is still visible, until
	 * `hasMore` turns false. Also wired to the "Retry" button shown for `error`.
	 */
	onLoadMore: () => void;
	/**
	 * A further page is in flight: blocks new requests and shows a spinner under
	 * the list. Pass the "loading more" state only — the first page of a list is
	 * the caller's to render, since it decides what an empty list looks like.
	 */
	isLoading: boolean;
	/** When false, nothing is watched and `onLoadMore` is never called again. */
	hasMore: boolean;
	/**
	 * Message for the page that just failed to load. While set, the sentinel
	 * isn't watched — the failed request stays on screen until the reader
	 * presses "Retry" instead of firing again on every intersection, which a
	 * sentinel still in view would otherwise do the instant loading clears.
	 * Pass the "loading more" error only, same as `isLoading` — a failed first
	 * page is the caller's to render, since `children` has nothing to keep
	 * showing underneath it.
	 */
	error?: string | null;
	/** How far below the fold to start loading. Defaults to 300px. */
	loadAheadPx?: number;
	/** Classes for the scroll container itself, e.g. padding. */
	className?: string;
	children: ReactNode;
};

/**
 * Scroll container that loads the next page as the reader approaches the end.
 *
 * The trigger is an empty element after the children, watched with an
 * `IntersectionObserver` rooted on this container — no scroll-offset
 * arithmetic, and nothing runs on scroll events that don't matter.
 */
export const InfiniteScroll = ({
	onLoadMore,
	isLoading,
	hasMore,
	error = null,
	loadAheadPx = 300,
	className,
	children,
}: InfiniteScrollProps) => {
	const rootRef = useRef<HTMLDivElement>(null);
	const sentinelRef = useRef<HTMLDivElement>(null);
	// Held in a ref so a caller passing an inline callback doesn't tear down and
	// re-create the observer on every render.
	const onLoadMoreRef = useRef(onLoadMore);

	useEffect(() => {
		onLoadMoreRef.current = onLoadMore;
	}, [onLoadMore]);

	useEffect(() => {
		const sentinel = sentinelRef.current;
		// Skipping the observer while a page loads is what stops one visible
		// sentinel from requesting the same page repeatedly: the effect re-runs
		// once loading ends, and only then asks again — which is wanted, since
		// a short page can leave the sentinel on screen. An `error` skips it the
		// same way, but doesn't clear on its own: nothing about the sentinel
		// changes after a failed request, so without this a still-visible
		// sentinel would retry immediately and forever on every failure.
		if (sentinel == null || !hasMore || isLoading || error != null) return undefined;

		const observer = new IntersectionObserver(
			(entries) => {
				if (entries.some((entry) => entry.isIntersecting)) onLoadMoreRef.current();
			},
			{ root: rootRef.current, rootMargin: `0px 0px ${loadAheadPx}px 0px` },
		);
		observer.observe(sentinel);
		return () => observer.disconnect();
	}, [hasMore, isLoading, error, loadAheadPx]);

	return (
		<div ref={rootRef} className={`overflow-y-auto ${className ?? ''}`}>
			{children}
			{hasMore && error == null && <div ref={sentinelRef} aria-hidden className="h-px" />}
			{isLoading && (
				<div className="flex justify-center py-4">
					<Spinner aria-label="Loading more" className="h-6 w-6" />
				</div>
			)}
			{error != null && (
				<div className="flex flex-col items-center gap-2 py-4">
					<p className="text-sm text-red-600 dark:text-red-300">{error}</p>
					<Button
						theme={ButtonTheme.Secondary}
						size={Size.SMALL}
						type="button"
						onClick={onLoadMore}
					>
						Retry
					</Button>
				</div>
			)}
		</div>
	);
};
