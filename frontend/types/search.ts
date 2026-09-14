// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { SearchObjectType } from '@/enums/search';

export type GlobalSearchBreadcrumb = {
	name: string;
	type: string;
	id?: string | null;
};

export type GlobalSearchItem = {
	id: string;
	name: string | null;
	/** Entity label (`Term`, `Table`, `ColumnAttribute`, …). Views stay `Table`. */
	type: SearchObjectType;
	/** Present on `Table` nodes; the UI maps view table types to the View tab. */
	table_type?: string | null;
	description: string | null;
	certified: boolean | string | null;
	parent_id: string | null;
	breadcrumbs: GlobalSearchBreadcrumb[];
	synonyms?: string[];
};

/**
 * What a query is matched against, and which kinds of object it may hit.
 *
 * `description` and `synonyms` widen the match past an object's name — its
 * description, and a Term's aliases. Both are omissible, and the backend's
 * defaults are not the same: `description` is off, `synonyms` on. See
 * `GlobalSearchFilters` in `gsf/server/search/router.py` for why.
 */
export type GlobalSearchFilters = {
	description?: boolean;
	synonyms?: boolean;
	objects?: SearchObjectType[];
};

export type GlobalSearchRequest = {
	search_term: string;
	text_match_option?: string;
	filters?: GlobalSearchFilters;
};
