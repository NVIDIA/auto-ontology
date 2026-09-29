// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { IconName } from '@/common/icons';
import { SearchObjectType } from '@/enums/search';
import type { GlobalSearchItem } from '@/types/search';

const VIEW_TABLE_TYPES = new Set(['view', 'materialized view']);

export const isViewTableType = (tableType: string | null | undefined): boolean =>
	VIEW_TABLE_TYPES.has((tableType ?? '').toLowerCase());

/**
 * The hit types a tag can be applied to, as `TAGGABLE_SEARCH_TYPES` in
 * `auto_ontology/server/rules/service.py` lists them.
 *
 * Raw `type` rather than the tab a hit is shown under: a view arrives as a
 * `Table` whose `table_type` says so and is labelled as one, which is why
 * `View` is deliberately absent. Databases and Schemas are containers a tag is
 * not applied to, and the two analysis kinds have no tag column at all.
 */
const TAGGABLE_SEARCH_TYPES = new Set<SearchObjectType>([
	SearchObjectType.Term,
	SearchObjectType.Table,
	SearchObjectType.Column,
	SearchObjectType.Attribute,
	SearchObjectType.SqlAttribute,
]);

/**
 * Whether a rule saved over a search would put its tags on this hit.
 *
 * Mirrors `_taggable_targets`: a kind that cannot carry a tag is dropped, and
 * so is a hit with no id, because a label needs something to point at. `id` is
 * typed as a string, but the search normalises a missing one to null rather
 * than leaving the row out.
 */
export const isTaggableSearchHit = (item: GlobalSearchItem): boolean =>
	TAGGABLE_SEARCH_TYPES.has(item.type) && Boolean(item.id);

/** UI type for a hit. Graph labels pass through; Table + view table_type → View. */
export const searchObjectTypeFromHit = (
	item: Pick<GlobalSearchItem, 'type' | 'table_type'>,
): SearchObjectType => {
	if (item.type === SearchObjectType.Table && isViewTableType(item.table_type)) {
		return SearchObjectType.View;
	}
	return item.type;
};

export const SEARCH_TYPE_LABEL: Record<SearchObjectType, string> = {
	[SearchObjectType.Term]: 'Term',
	[SearchObjectType.Attribute]: 'Column Attribute',
	[SearchObjectType.SqlAttribute]: 'SQL Attribute',
	[SearchObjectType.Analysis]: 'Analysis',
	[SearchObjectType.PqlAnalysis]: 'PQL Analysis',
	[SearchObjectType.Db]: 'Database',
	[SearchObjectType.Schema]: 'Schema',
	[SearchObjectType.Table]: 'Table',
	[SearchObjectType.View]: 'View',
	[SearchObjectType.Column]: 'Column',
};

export const SEARCH_TYPE_TAB_LABEL: Record<SearchObjectType, string> = {
	[SearchObjectType.Term]: 'Terms',
	[SearchObjectType.Attribute]: 'Column Attributes',
	[SearchObjectType.SqlAttribute]: 'SQL Attributes',
	[SearchObjectType.Analysis]: 'Analyses',
	[SearchObjectType.PqlAnalysis]: 'PQL Analyses',
	[SearchObjectType.Db]: 'Databases',
	[SearchObjectType.Schema]: 'Schemas',
	[SearchObjectType.Table]: 'Tables',
	[SearchObjectType.View]: 'Views',
	[SearchObjectType.Column]: 'Columns',
};

export const SEARCH_TYPE_ICON: Record<SearchObjectType, IconName> = {
	[SearchObjectType.Term]: IconName.Terms,
	[SearchObjectType.Attribute]: IconName.Key,
	[SearchObjectType.SqlAttribute]: IconName.CodeBracket,
	[SearchObjectType.Analysis]: IconName.ChartBar,
	[SearchObjectType.PqlAnalysis]: IconName.ChartLine,
	[SearchObjectType.Db]: IconName.Database,
	[SearchObjectType.Schema]: IconName.Schema,
	[SearchObjectType.Table]: IconName.Table,
	[SearchObjectType.View]: IconName.View,
	[SearchObjectType.Column]: IconName.Column,
};

export const SEARCH_TYPE_TAB_ORDER: SearchObjectType[] = [
	SearchObjectType.Term,
	SearchObjectType.Attribute,
	SearchObjectType.SqlAttribute,
	SearchObjectType.Analysis,
	SearchObjectType.PqlAnalysis,
	SearchObjectType.Db,
	SearchObjectType.Schema,
	SearchObjectType.Table,
	SearchObjectType.View,
	SearchObjectType.Column,
];
