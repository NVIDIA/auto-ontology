// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** Graph labels used as search types. `View` is Table + a view table_type. */
export enum SearchObjectType {
	Term = 'Term',
	Attribute = 'ColumnAttribute',
	SqlAttribute = 'SqlAttribute',
	Analysis = 'CustomAnalysis',
	PqlAnalysis = 'PqlAnalysis',
	Db = 'Database',
	Schema = 'Schema',
	Table = 'Table',
	View = 'View',
	Column = 'Column',
	/**
	 * A tag itself, not the objects carrying it — `LABEL_TAG` in
	 * `auto_ontology/dal/search.py`. Matched on its own name, and the only kind
	 * with no description.
	 */
	Tag = 'Tag',
}

export enum TextMatchOption {
	Contains = 'contains',
}
