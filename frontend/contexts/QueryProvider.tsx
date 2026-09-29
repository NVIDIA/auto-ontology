// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { ReactNode } from 'react';

const makeQueryClient = (): QueryClient =>
	new QueryClient({
		defaultOptions: {
			queries: {
				/**
				 * Data handed over by a server render would otherwise be
				 * refetched the moment it reached the browser: the default of
				 * 0 marks it stale on arrival, which would pay for the
				 * prefetch twice. A minute covers opening a page and the
				 * navigations that follow it — a write doesn't wait it out,
				 * since it invalidates its own key.
				 */
				staleTime: 60_000,
				/**
				 * Every read behind a query is permission-gated, and a refused
				 * one is refused the same way three times. One retry covers a
				 * dropped connection without turning a 403 into four requests.
				 */
				retry: 1,
			},
		},
	});

let browserQueryClient: QueryClient | undefined;

const getQueryClient = (): QueryClient => {
	// A client per server render, so one request's data is never served to the
	// next request's user. The browser keeps one for the life of the tab —
	// that cache *is* the shared store, and remaking it would empty it.
	if (typeof window === 'undefined') return makeQueryClient();
	browserQueryClient ??= makeQueryClient();
	return browserQueryClient;
};

/**
 * The browser-side cache every `useQuery` in the app reads from.
 *
 * Mounted at the root so a list read on one page is not read again on the
 * next: the query key is what makes two components asking for the same thing
 * one request. Server components hand their reads to it through
 * `HydrationBoundary`, so the first render already holds them.
 */
export const QueryProvider = ({ children }: { children: ReactNode }) => (
	<QueryClientProvider client={getQueryClient()}>{children}</QueryClientProvider>
);
