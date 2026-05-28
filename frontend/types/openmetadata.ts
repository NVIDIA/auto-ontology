// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * TypeScript shapes for the subset of the OpenMetadata API we use in the
 * /catalog demo pages. Mirrors what we observed in openmetadata-eval/evidence/.
 *
 * We deliberately keep these narrow — only fields the UI reads. OM is happy to
 * return many more fields when ?fields=... is set.
 */

export type OmTagState = 'Suggested' | 'Confirmed';
export type OmTagSource = 'Classification' | 'Glossary';
export type OmTagLabelType = 'Manual' | 'Propagated' | 'Automated' | 'Derived' | 'Generated';

export type OmTagLabel = {
	tagFQN: string;
	source: OmTagSource;
	state: OmTagState;
	labelType: OmTagLabelType;
	description?: string | null;
	name?: string;
};

export type OmEntityRef = {
	id: string;
	type: string;
	name: string;
	fullyQualifiedName?: string;
	displayName?: string | null;
	description?: string | null;
	deleted?: boolean;
	href?: string;
};

export type OmColumn = {
	name: string;
	displayName?: string | null;
	dataType: string;
	dataTypeDisplay?: string;
	description?: string | null;
	tags?: OmTagLabel[];
	constraint?: string | null;
	ordinalPosition?: number;
	fullyQualifiedName?: string;
};

export type OmUsageSummary = {
	dailyStats?: { count: number; percentileRank?: number };
	weeklyStats?: { count: number; percentileRank?: number };
	monthlyStats?: { count: number; percentileRank?: number };
	date?: string;
} | null;

export type OmTable = {
	id: string;
	name: string;
	displayName?: string | null;
	fullyQualifiedName: string;
	description?: string | null;
	tableType?: string;
	columns: OmColumn[];
	tags?: OmTagLabel[];
	owners?: OmEntityRef[] | null;
	database: OmEntityRef;
	databaseSchema: OmEntityRef;
	service?: OmEntityRef;
	usageSummary?: OmUsageSummary;
	updatedAt?: number;
	version?: number;
};

export type OmTagDef = {
	id?: string;
	name: string;
	displayName?: string | null;
	description?: string | null;
	fullyQualifiedName: string;
	classification: OmEntityRef;
	mutuallyExclusive?: boolean;
	deprecated?: boolean;
};

export type OmClassification = {
	id?: string;
	name: string;
	displayName?: string | null;
	description?: string | null;
	provider?: 'system' | 'user';
	mutuallyExclusive?: boolean;
	termCount?: number;
	fullyQualifiedName?: string;
};

export type OmQuery = {
	id: string;
	name?: string;
	query: string;
	queryType?: string | null;
	queryDate?: number;
	users?: OmEntityRef[] | null;
	queryUsedIn?: OmEntityRef[] | null;
	vote?: number | null;
	exclude_usage?: boolean | null;
	usedBy?: string[] | null;
	processedLineage?: boolean;
};

export type OmListResponse<T> = {
	data: T[];
	paging?: {
		total?: number;
		offset?: number;
		limit?: number;
		after?: string;
		before?: string;
	};
};
