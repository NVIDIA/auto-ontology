// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ComposerColumnType, ComposerSectionKind } from '@/enums/datasources';

/** Certification state attached to a certifiable field (name / description). */
export type ComposerCertification = {
	certified: boolean;
	/** Renders the full pill (icon + status text) instead of the icon-only badge. */
	showLabel?: boolean;
};

export type ComposerTextCardSection = {
	type: ComposerSectionKind.TEXT_CARD;
	id: string;
	title: string;
	body: string;
	editable?: boolean;
	suggestable?: boolean;
	/** When set, renders a certification badge (view) / dropdown (edit) on the card. */
	certification?: ComposerCertification;
};

export type ComposerTagListSection = {
	type: ComposerSectionKind.TAG_LIST;
	id: string;
	title: string;
	values: string[];
	editable?: boolean;
	/** Optional hint shown under the input while editing. */
	hint?: string;
};

export type ComposerInfoGridSection = {
	type: ComposerSectionKind.INFO_GRID;
	id: string;
	title: string;
	items: { label: string; value: string }[];
};

type ComposerDataTableColumnBase = {
	key: string;
	label: string;
	/** Horizontal alignment of the header and cell content. Defaults to `left`. */
	align?: 'left' | 'center' | 'right';
	/** Tailwind width class for the column (e.g. `w-44`). Best paired with a fixed table `layout`. */
	width?: string;
};

/**
 * Clipping is offered on text columns only. Pills clip and reveal themselves
 * chip by chip, and a badge is an icon, so a cell-wide tooltip over either
 * would be a second tooltip on top of the value's own one.
 */
export type ComposerDataTableColumn =
	| (ComposerDataTableColumnBase & {
			/** Renders the cell as plain text. The default when omitted. */
			type?: ComposerColumnType.TEXT;
			/** Clips long text to one line and shows the full value in a popover on hover, only when clipped. */
			truncate?: boolean;
			/**
			 * Tailwind max-width class applied to the cell when `truncate` is set.
			 * Defaults to `max-w-0`, which lets the line fill the whole column.
			 */
			maxWidthClass?: string;
	  })
	| (ComposerDataTableColumnBase & {
			/**
			 * Renders the cell as a list of read-only pills (`TAGS`) or an icon-only
			 * certification badge (`CERTIFICATION`, cell value is a
			 * `CertificationStatus`) instead of plain text.
			 */
			type: ComposerColumnType.TAGS | ComposerColumnType.CERTIFICATION;
	  });

export type ComposerDataTableSection = {
	type: ComposerSectionKind.DATA_TABLE;
	id: string;
	title: string;
	columns: ComposerDataTableColumn[];
	rows: Record<string, string | string[]>[];
	/** When set with `onDataTableRowClick`, rows become clickable using this field as id. */
	rowIdKey?: string;
	/** Shown instead of the table (columns included) when `rows` is empty. */
	emptyMessage?: string;
	/** Column sizing strategy. Defaults to `auto`. Use `fixed` with column `width`s to align tables. */
	layout?: 'fixed' | 'auto';
};

export type ComposerLoadingPanelSection = {
	type: ComposerSectionKind.LOADING_PANEL;
	id: string;
	/** Shown under the spinner. */
	message: string;
};

export type ComposerZoneChip = {
	id: string;
	name: string;
	color: string | null;
	enabled: boolean;
};

export type ComposerZonesSection = {
	type: ComposerSectionKind.ZONES_CHIPS;
	id: string;
	title: string;
	zones: ComposerZoneChip[];
};

export type ComposerRelatedTermChip = {
	id: string;
	name: string;
};

export type ComposerRelatedTermsSection = {
	type: ComposerSectionKind.RELATED_TERMS_CHIPS;
	id: string;
	title: string;
	terms: ComposerRelatedTermChip[];
};

export type ComposerEntity = {
	id: string;
	name: string;
	/** Catalog focus path (`dbId|schemaId|tableId` or `…|columnId`) used to navigate. */
	focusId: string;
};

export type ComposerEntitiesSection = {
	type: ComposerSectionKind.ENTITY_CHIPS;
	id: string;
	title: string;
	entities: ComposerEntity[];
};

export type ComposerSqlBlockSection = {
	type: ComposerSectionKind.SQL_BLOCK;
	id: string;
	title: string;
	sql: string;
	editable?: boolean;
};

export type ComposerSection =
	| ComposerTextCardSection
	| ComposerTagListSection
	| ComposerInfoGridSection
	| ComposerDataTableSection
	| ComposerLoadingPanelSection
	| ComposerZonesSection
	| ComposerRelatedTermsSection
	| ComposerEntitiesSection
	| ComposerSqlBlockSection;

const composerSectionTypes: readonly ComposerSectionKind[] = [
	ComposerSectionKind.TEXT_CARD,
	ComposerSectionKind.TAG_LIST,
	ComposerSectionKind.INFO_GRID,
	ComposerSectionKind.DATA_TABLE,
	ComposerSectionKind.LOADING_PANEL,
	ComposerSectionKind.ZONES_CHIPS,
	ComposerSectionKind.RELATED_TERMS_CHIPS,
	ComposerSectionKind.ENTITY_CHIPS,
	ComposerSectionKind.SQL_BLOCK,
];

export function isComposerSection(x: unknown): x is ComposerSection {
	return (
		x !== null &&
		typeof x === 'object' &&
		'type' in x &&
		composerSectionTypes.includes((x as ComposerSection).type as ComposerSectionKind)
	);
}
