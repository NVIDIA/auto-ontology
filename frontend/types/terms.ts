// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { CertificationStatus } from '@/enums/certification';

export type TermZone = {
	id: string;
	name: string;
	color: string | null;
	enabled: boolean;
};

export type Term = {
	id: string;
	name: string;
	description: string | null;
	synonyms: string[];
	zones: TermZone[];
	name_certified: boolean;
	description_certified: boolean;
	/**
	 * Aggregate three-state certification status: the term's own
	 * name/description flags plus every column & sql attribute flag, rolled up
	 * server-side (see `_certification_flags_clause` in gsf/dal/terms.py) and
	 * zone-scoped to the same boundary as the attribute list endpoints. Both
	 * `/terms` and `/terms/{id}` return it, and every certification PATCH
	 * returns the recomputed value — never derive it on the client.
	 */
	certification: CertificationStatus;
};

export type TermTable = {
	id: string;
	name: string;
	schema_id: string;
	db_id: string;
};

export type TermDetail = Term & {
	table_count: number;
	tables: TermTable[];
	related_terms: RelatedTerm[];
};

export type AttributeColumnRef = {
	id: string;
	column_name: string;
	table_id: string;
	table_name: string;
	schema_id: string;
	db_id: string;
};

export type ColumnAttribute = {
	id: string;
	name: string;
	description: string | null;
	term_name: string;
	source_column: string;
	datatype: string | null;
	table_id: string;
	/** Profiled sample values from the owning Column, when available. */
	sample_values: string[] | null;
	zones: TermZone[];
	/** The Column that owns this attribute via `HAS_ATTRIBUTE`, if any. */
	primary_column: AttributeColumnRef | null;
	/** Columns elsewhere that point at this attribute via `SEMANTIC_FK`. */
	referenced_columns: AttributeColumnRef[];
	certified: boolean;
};

export type SqlAttribute = {
	id: string;
	name: string;
	description: string | null;
	expression: string | null;
	source: string | null;
	sql: string | null;
	term_id: string;
	term_name: string;
	zones?: TermZone[];
	certified: boolean;
};

export type RelatedTerm = {
	id: string;
	name: string;
	description: string | null;
};

export type TermCount = {
	term_id: string;
	count: number;
};

export type RelatedTermCount = TermCount;
