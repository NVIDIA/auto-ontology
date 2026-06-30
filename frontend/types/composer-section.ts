// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ComposerSectionKind } from '@/enums/datasources';

export type ComposerTextCardSection = {
	type: ComposerSectionKind.TEXT_CARD;
	id: string;
	title: string;
	body: string;
	editable?: boolean;
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

export type ComposerDataTableSection = {
	type: ComposerSectionKind.DATA_TABLE;
	id: string;
	title: string;
	columns: { key: string; label: string }[];
	rows: Record<string, string>[];
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
	description: string | null;
};

export type ComposerRelatedTermsSection = {
	type: ComposerSectionKind.RELATED_TERMS_CHIPS;
	id: string;
	title: string;
	terms: ComposerRelatedTermChip[];
};

export type ComposerSection =
	| ComposerTextCardSection
	| ComposerTagListSection
	| ComposerInfoGridSection
	| ComposerDataTableSection
	| ComposerLoadingPanelSection
	| ComposerZonesSection
	| ComposerRelatedTermsSection;

const composerSectionTypes: readonly ComposerSectionKind[] = [
	ComposerSectionKind.TEXT_CARD,
	ComposerSectionKind.TAG_LIST,
	ComposerSectionKind.INFO_GRID,
	ComposerSectionKind.DATA_TABLE,
	ComposerSectionKind.LOADING_PANEL,
	ComposerSectionKind.ZONES_CHIPS,
	ComposerSectionKind.RELATED_TERMS_CHIPS,
];

export function isComposerSection(x: unknown): x is ComposerSection {
	return (
		x !== null &&
		typeof x === 'object' &&
		'type' in x &&
		composerSectionTypes.includes((x as ComposerSection).type as ComposerSectionKind)
	);
}
