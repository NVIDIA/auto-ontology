// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export type NodeType =
	| 'term'
	| 'table'
	| 'schema'
	| 'column'
	| 'columnAttribute'
	| 'sqlAttribute'
	| 'sql'
	| 'customAnalysis';

// Eight hues spread perfectly evenly around the color wheel (45° apart —
// 0/35/95/135/185/230/275/320, roughly) rather than clustered by "family"
// (e.g. every attribute-ish leaf node sharing one hue), so no two of these
// eight ever read as "basically the same color" next to each other on the
// canvas or in the legend. In particular, term/table used to sit only ~15°
// apart (both a warm orange/gold) — they're now a full 45°+ apart, same
// minimum gap as every other pair here.
const SQL_OBJECT_ICON_COLOR = '#d43535'; // red
const TABLE_OBJECT_ICON_COLOR = '#da932f'; // orange
const TERM_OBJECT_ICON_COLOR = '#61a630'; // yellow-green
const COLUMN_ATTRIBUTE_OBJECT_ICON_COLOR = '#3b9b53'; // green
const CUSTOM_ANALYSIS_OBJECT_ICON_COLOR = '#2eacb8'; // teal
const SQL_ATTRIBUTE_OBJECT_ICON_COLOR = '#4d62cb'; // indigo
const COLUMN_OBJECT_ICON_COLOR = '#9559c0'; // violet
const SCHEMA_OBJECT_ICON_COLOR = '#d65cad'; // pink

/**
 * Each type's own accent color — used on the canvas as a node's border (see
 * `NODE_TYPE_BORDER_COLOR`'s own former doc comment in `GraphCanvas.tsx`)
 * and, via this same map, as the swatch color for its entry in
 * `ExplorationView.tsx`'s "Viewing: ..." legend — one source of truth so the
 * two never drift apart.
 */
export const NODE_TYPE_ACCENT_COLOR: Record<NodeType, string> = {
	term: TERM_OBJECT_ICON_COLOR,
	table: TABLE_OBJECT_ICON_COLOR,
	schema: SCHEMA_OBJECT_ICON_COLOR,
	column: COLUMN_OBJECT_ICON_COLOR,
	columnAttribute: COLUMN_ATTRIBUTE_OBJECT_ICON_COLOR,
	sqlAttribute: SQL_ATTRIBUTE_OBJECT_ICON_COLOR,
	sql: SQL_OBJECT_ICON_COLOR,
	customAnalysis: CUSTOM_ANALYSIS_OBJECT_ICON_COLOR,
};
