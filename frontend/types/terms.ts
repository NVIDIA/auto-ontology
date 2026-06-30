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

export type TermDetail = Term & {
	table_count: number;
	zones: TermZone[];
};

export type TermAttribute = {
	id: string;
	name: string;
	description: string | null;
	term_name: string;
	source_column: string;
	datatype: string | null;
	table_id: string;
	fk_count: number;
	is_primary_key: boolean;
};
