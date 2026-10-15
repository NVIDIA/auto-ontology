// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { CertificationStatus } from '@/enums/certification';
import type { TagAuthor, TagChip } from '@/types/tags';

/** Who last saved a glossary row, once the detail read has resolved the account. */
export type GlossaryActivity = {
	/** ISO 8601. Null for a row that predates the column. */
	created?: string | null;
	/** ISO 8601. Null until someone saves an edit. */
	modified?: string | null;
	created_by_user?: TagAuthor | null;
	modified_by_user?: TagAuthor | null;
};

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
	 * server-side (see `_certification_flags_clause` in auto_ontology/dal/terms.py) and
	 * zone-scoped to the same boundary as the attribute list endpoints. Both
	 * `/terms` and `/terms/{id}` return it, and every certification PATCH
	 * returns the recomputed value — never derive it on the client.
	 */
	certification: CertificationStatus;
	/** A user id or a group id. Null when the term is unassigned. */
	owner_id?: string | null;
	/**
	 * Chips on the list card. The paged list resolves them with the page.
	 * Absent only on a response that predates that, which the card treats as
	 * no tags.
	 */
	tags?: TagChip[];
};

export type TermTable = {
	id: string;
	name: string;
	schema_id: string;
	db_id: string;
};

/** The detail read adds the tables and related terms the list card does not. */
export type TermDetail = Term &
	GlossaryActivity & {
		table_count: number;
		tables: TermTable[];
		related_terms: RelatedTerm[];
		tags: TagChip[];
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
	/** Absent from the patch response, which echoes only the fields it wrote. */
	tags?: TagChip[];
	/** The Column that owns this attribute via `HAS_ATTRIBUTE`, if any. */
	primary_column: AttributeColumnRef | null;
	/** Columns elsewhere that point at this attribute via `SEMANTIC_FK`. */
	referenced_columns: AttributeColumnRef[];
	certified: boolean;
} & GlossaryActivity;

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
	tags?: TagChip[];
	certified: boolean;
	/** A user id or a group id. Null when the attribute is unassigned. */
	owner_id?: string | null;
} & GlossaryActivity;

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
