// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export enum DataModels {
	DB = 'db',
	SCHEMA = 'schema',
	TABLE = 'base table',
	VIEW = 'view',
	MATERIALIZED_VIEW = 'materialized view',
	COLUMN = 'column',
}

/** Postgres ``table_type`` values stored on catalog tables. */
export enum TableType {
	BASE_TABLE = 'base table',
	VIEW = 'view',
	MATERIALIZED_VIEW = 'materialized view',
}

/** Tree focus resolution before a catalog entity is matched. */
export enum TreeFocusState {
	NONE = 'none',
	LOADING = 'loading',
}

/** Block type in SinglePageComposer sections. */
export enum ComposerSectionKind {
	TEXT_CARD = 'textCard',
	TAG_LIST = 'tagList',
	INFO_GRID = 'infoGrid',
	DATA_TABLE = 'dataTable',
	LOADING_PANEL = 'loadingPanel',
	ZONES_CHIPS = 'zonesChips',
	RELATED_TERMS_CHIPS = 'relatedTermsChips',
	ENTITY_CHIPS = 'entityChips',
	/**
	 * Tags the object carries, each removable, with a picker to add another.
	 * Not to be confused with `TAG_LIST`, which is a list of plain strings a
	 * field happens to hold (synonyms, sample values) and is edited as text.
	 */
	ENTITY_TAGS = 'entityTags',
	SQL_BLOCK = 'sqlBlock',
}

/** How a DATA_TABLE column renders its cells. */
export enum ComposerColumnType {
	TEXT = 'text',
	TAGS = 'tags',
	CERTIFICATION = 'certification',
}

export enum Usage {
	HIGH = 'high',
	MEDIUM = 'medium',
	LOW = 'low',
	UNUSED = 'unused',
}
