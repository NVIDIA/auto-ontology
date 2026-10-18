// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { DiscoveryFilters } from '@/types/discovery';

/**
 * What Discovery searches with until somebody narrows it.
 *
 * Description matching on, which is what `DEFAULT_SEARCH_FILTERS` in
 * `common/GlobalSearch.tsx` runs with rather than what the backend defaults to
 * — the same query has to find the same things whether it was typed here or in
 * the top bar's dialog. Synonym matching is on for both too, but nothing here
 * turns it off, so it is sent as a constant instead of held as a filter.
 */
export const DEFAULT_DISCOVERY_FILTERS: DiscoveryFilters = {
	// Null rather than every option listed, which is also what makes this a
	// usable default for the tags: the vocabulary is read over the network, so
	// there is no list of them to spell out at module scope.
	objects: null,
	description: true,
	tags: null,
	data: null,
};
