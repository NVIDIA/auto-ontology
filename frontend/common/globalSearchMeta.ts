// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { IconName } from '@/common/icons';
import { SearchObjectType } from '@/enums/search';
import type { GlobalSearchItem } from '@/types/search';

const VIEW_TABLE_TYPES = new Set(['view', 'materialized view']);

export const isViewTableType = (tableType: string | null | undefined): boolean =>
	VIEW_TABLE_TYPES.has((tableType ?? '').toLowerCase());

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
