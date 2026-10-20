// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { SearchObjectType } from '@/enums/search';

/**
 * The Discovery panel's filters, as the screen holds them.
 *
 * One field per thing `GlobalSearchFilters` can carry, and nothing else: the
 * search behind this page is the global one, so a filter it cannot send would
 * narrow the screen without narrowing the query — the results would disagree
 * with the counts behind the tabs, which are taken by the same request.
 */
export type DiscoveryFilters = {
	/**
	 * Kinds a hit may be.
	 *
	 * Null is every kind — the `undefined` the request sends, rather than all
	 * eleven spelled out. An *empty array* is the opposite and is reachable:
	 * Deselect All leaves no kind ticked, and a search with no kind to look in
	 * finds nothing. The two have to be told apart, which is the whole reason
	 * this is nullable instead of using empty for "all".
	 */
	objects: SearchObjectType[] | null;
	/** Match an object's description as well as its name. */
	description: boolean;
	/**
	 * Tag ids a hit may carry, plus `UNTAGGED_TAG_FILTER` for the objects
	 * carrying none.
	 *
	 * Null and empty read as they do for `objects`: null asks nothing about
	 * tags, empty is Deselect All and matches nothing. Neither is "objects
	 * with no tags" — that third reading is what the sentinel is for.
	 */
	tags: string[] | null;
	/**
	 * Database and schema ids a hit must live under, in one list.
	 *
	 * One list rather than two because the tree they are ticked in is one
	 * tree: a whole database and a schema picked out of another are a single
	 * selection. A ticked database stays one id instead of being expanded
	 * into its schemas, so it keeps meaning "this database" as schemas are
	 * added to it.
	 *
	 * Null and empty read as they do for `objects`. Unlike the other two,
	 * a non-empty selection also narrows the *kinds* returned: only tables,
	 * views and columns live under a schema, so nothing else can match.
	 */
	data: string[] | null;
};

/**
 * A search the screen has run, as opposed to the one being typed.
 *
 * The two are held apart because this page searches on Apply rather than on
 * every keystroke: the results, the tab counts and any rule saved over them all
 * belong to the term and filters that were applied, not to whatever the panel
 * has been changed to since.
 */
export type DiscoverySearch = {
	term: string;
	filters: DiscoveryFilters;
};
