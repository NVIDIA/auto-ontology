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

export type GlobalSearchFilters = {
	description?: boolean;
	objects?: SearchObjectType[];
};

export type GlobalSearchRequest = {
	search_term: string;
	text_match_option?: string;
	filters?: GlobalSearchFilters;
};
