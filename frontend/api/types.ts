// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Column, Schema } from '@/types/datasources';

export type ApiError = {
	message: string;
	error: boolean;
};

export type ResponseWithError<T> = T & Partial<ApiError>;

export type ResponseWithCount<T> = {
	data: T;
	count: number;
};

export type ApiResponse<T> = ResponseWithError<ResponseWithCount<T>>;

/** `ApiResponse` for one page: `count` is this page, `total` the whole list. */
export type ApiPagedResponse<T> = ResponseWithError<ResponseWithCount<T> & { total: number }>;

/**
 * Window into a list endpoint. Omit `limit` for the whole list — the backend
 * caps a requested page at 100 rows.
 */
export type PageParams = {
	skip?: number;
	limit?: number;
};

/** Schemas endpoint returns a non-standard envelope (not {data,count}). */
export type SchemasResponse = {
	schemas_count: number;
	schemas: Omit<Schema, 'tables'>[];
};

/** Payload accepted by PATCH /nodes/:id — all fields optional. */
export type NodePatch = {
	description?: string;
	sample_values?: string[];
	description_certified?: boolean;
};

/** Response from the node update endpoint. */
export type NodeUpdateResponse = {
	id: string;
} & NodePatch;

/** Columns endpoint returns a table-scoped envelope with nested column rows. */
export type ColumnsEnvelope = {
	table_name: string;
	schema_name: string;
	database_name: string;
	columns_count: number;
	columns: Pick<
		Column,
		| 'id'
		| 'ordinal_position'
		| 'column_name'
		| 'data_type'
		| 'description'
		| 'description_certified'
		| 'sample_values'
	>[];
};
