// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useCallback, useMemo, useState } from 'react';

import type { TablePagination } from '@/types/table';

export const DEFAULT_PAGE_SIZE = 10;

export type UsePaginationOptions = {
	/**
	 * Rows across all pages: the length of the array being paged when it is
	 * held locally, or the count the server reported when it isn't.
	 */
	totalItems: number;
	/** Rows per page. Defaults to `DEFAULT_PAGE_SIZE`. */
	pageSize?: number;
	/**
	 * What the pages belong to — the term a modal was opened on, say. When it
	 * changes the paging starts over, since the page the previous object was
	 * left on says nothing about this one. It is derived rather than pushed
	 * through an effect, so switching costs no extra render and, for a list read
	 * from a server, no request for a page that is about to be abandoned.
	 */
	resetKey?: string | null;
};

export type UsePaginationResult = {
	/** 1-based current page, never past the last one the rows reach. */
	page: number;
	pageSize: number;
	/** Rows before the current page: it covers `[skip, skip + pageSize)`. */
	skip: number;
	/**
	 * Rows the current page holds — `pageSize`, or fewer on a last page that
	 * doesn't fill up. Sizes a loading placeholder to the rows it stands in for,
	 * so the list doesn't resize once they arrive. Falls back to `pageSize`
	 * while the total is still unknown.
	 */
	pageRowCount: number;
	setPage: (page: number) => void;
	/** Ready to hand to `Table`'s `pagination` prop. */
	pagination: TablePagination;
};

/**
 * Keeps the position within a paged list: which page is current, how wide a
 * page is, and where that page starts.
 *
 * The rows themselves stay with the caller, which is what lets one hook serve
 * both cases — slice a local array by `skip`, or read `[skip, skip + pageSize)`
 * from a server and report what it says the total is.
 */
export function usePagination({
	totalItems,
	pageSize = DEFAULT_PAGE_SIZE,
	resetKey = null,
}: UsePaginationOptions): UsePaginationResult {
	const [paged, setPaged] = useState({ key: resetKey, page: 1 });

	const pageCount = Math.max(1, Math.ceil(totalItems / pageSize));
	// Rows can also shrink under a page that is still current — a list narrowed
	// by a filter — which leaves the stored page out of range.
	const page = Math.min(paged.key === resetKey ? paged.page : 1, pageCount);

	// Store what is being read. Leaving the clamp out of state means the two
	// disagree, and the moment the rows come back the position springs back to
	// the page nobody is on — asking the server for it on the way. Writing
	// during render rather than from an effect keeps `skip` and the stored page
	// in step within one commit, so no request goes out from the state between.
	if (paged.key !== resetKey || paged.page !== page) {
		setPaged({ key: resetKey, page });
	}

	const setPage = useCallback(
		(next: number) => setPaged({ key: resetKey, page: next }),
		[resetKey],
	);

	const pagination = useMemo<TablePagination>(
		() => ({ page, pageSize, totalItems, onPageChange: setPage }),
		[page, pageSize, totalItems, setPage],
	);

	const skip = (page - 1) * pageSize;

	return {
		page,
		pageSize,
		skip,
		pageRowCount:
			totalItems > 0 ? Math.max(1, Math.min(pageSize, totalItems - skip)) : pageSize,
		setPage,
		pagination,
	};
}
