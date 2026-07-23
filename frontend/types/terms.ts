// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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
