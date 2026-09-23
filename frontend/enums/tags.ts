// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Which of the five things a tag points at.
 *
 * The values are the backend's — see the `TARGET_*` constants in
 * `auto_ontology/dal/tags.py`, which is what puts them in the `type` field, and the
 * `TagTargetType` enum its attach and detach routes validate against.
 */
export enum TagItemType {
	Term = 'term',
	Table = 'table',
	Column = 'column',
	ColumnAttribute = 'column_attribute',
	SqlAttribute = 'sql_attribute',
}
