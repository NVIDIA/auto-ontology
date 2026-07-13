// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export type Term = {
	id: string;
	name: string;
	description: string | null;
	synonyms: string[];
};

export type TermZone = {
	id: string;
	name: string;
	color: string | null;
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
	zones: TermZone[];
};

export type ColumnAttribute = {
	id: string;
	name: string;
	description: string | null;
	term_name: string;
	source_column: string;
	datatype: string | null;
	table_id: string;
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

export type RelatedTermCount = {
	term_id: string;
	count: number;
};
