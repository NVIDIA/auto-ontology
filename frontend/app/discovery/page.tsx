// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DiscoveryView } from '@/components/discoveryPage';

/**
 * `?q=` is how the top bar's search dialog hands a search over when somebody
 * asks for Advanced Search, so the page opens on the results they were already
 * looking at.
 *
 * Read here and passed down rather than through `useSearchParams` in the view:
 * it is only ever an opening value, and a hook would have the view re-read it
 * on every navigation and need a Suspense boundary to do so. Repeating the
 * parameter yields an array, which is nobody's search and is dropped.
 *
 * `?n=` is a one-off token the dialog adds to each hand-over. It is not part of
 * any search: it exists so that handing over the term already on screen is a
 * different URL, and therefore a navigation the view can tell apart.
 */
export default async function DiscoveryPage({
	searchParams,
}: {
	searchParams: Promise<{ q?: string | string[]; n?: string | string[] }>;
}) {
	const { q, n } = await searchParams;
	return (
		<DiscoveryView
			initialQuery={typeof q === 'string' ? q : ''}
			handoverId={typeof n === 'string' ? n : ''}
		/>
	);
}
