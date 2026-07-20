// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { TableType } from '@/enums/datasources';

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
