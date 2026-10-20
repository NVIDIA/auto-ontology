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
 * `GlobalSearchFilters` in `auto_ontology/server/search/router.py` for why.
 *
 * `objects`, `tags` and `data` narrow instead, and omitting any of them
 * narrows by nothing.
 */
export type GlobalSearchFilters = {
	description?: boolean;
	synonyms?: boolean;
	objects?: SearchObjectType[];
	/**
	 * Tag ids a hit may carry, plus `UNTAGGED_TAG_FILTER` for the objects
	 * carrying none. Any one entry is enough, so two tags widen the selection
	 * within the narrowing.
	 *
	 * Only five kinds can be tagged, so this also drops Databases, Schemas and
	 * both analysis kinds — unless the untagged sentinel is present, since a
	 * kind that cannot be tagged is untagged.
	 */
	tags?: string[];
	/**
	 * Database and schema ids in one list. A hit matches when its own schema,
	 * or that schema's database, is among them.
	 *
	 * Narrows the kinds too, and harder than `tags` does: only Tables, Views
	 * and Columns live under a schema, so everything else is dropped — a Term
	 * reached by an alias included.
	 */
	data?: string[];
};

export type GlobalSearchRequest = {
	search_term: string;
	text_match_option?: string;
	filters?: GlobalSearchFilters;
};
