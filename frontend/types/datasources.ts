// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { TableType } from '@/enums/datasources';
import type { TagChip } from '@/types/tags';

export type Column = {
	id: string;
	column_name: string;
	data_type: string;
	database_name: string;
	schema_name: string;
	table_name: string;
	ordinal_position: number;
	description?: string;
	sample_values?: string[];
	description_certified?: boolean;
	/** Absent until the table's columns have been read. */
	tags?: TagChip[];
};

export type Table = {
	id: string;
	name: string;
	database_name: string;
	schema_name: string;
	columns_count: number;
	sql_count?: number;
	terms_count?: number;
	columns: Column[];
	table_type: TableType;
	description?: string;
	description_certified?: boolean;
	/**
	 * Read with the schema's table list, since the catalog has no table detail
	 * endpoint of its own. Absent on a table the tree only knows a count for.
	 */
	tags?: TagChip[];
};

export type Schema = {
	id: string;
	schema_name: string;
	tables_count: number;
	tables: Table[];
	description?: string;
};

export type Database = {
	id: string;
	name: string;
	num_of_schemas: number;
	schemas: Schema[];
	description?: string;
};
