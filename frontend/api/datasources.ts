// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { pageQuery, requests } from './requests';
import type { Column, Database, Schema, Table } from '@/types/datasources';
import type { Params } from '@/types/params';
import type {
	ApiPagedResponse,
	ApiResponse,
	ColumnsEnvelope,
	NodePatch,
	NodeUpdateResponse,
	PageParams,
	ResponseWithCount,
	SchemasResponse,
} from './types';

/** What the columns endpoint sends: the envelope, plus the table's full count. */
type ColumnsPageResult = ResponseWithCount<ColumnsEnvelope> & { total: number };

// ---------------------------------------------------------------------------
// Request dedup caches
// ---------------------------------------------------------------------------

const schemasByDbMap = new Map<string, Promise<ApiResponse<Schema[]>>>();
const tablesBySchemaMap = new Map<string, Promise<ApiResponse<Table[]>>>();
const columnsByTableMap = new Map<string, Promise<ApiPagedResponse<Column[]>>>();

export const datasources = {
	getDBs: () => requests.get<ResponseWithCount<Database[]>>('datasources/dbs'),

	/** Schemas for one database; parallel callers with the same key share one HTTP request. */
	getSchemasForDatabase: (dbId: string): Promise<ApiResponse<Schema[]>> => {
		const pending = schemasByDbMap.get(dbId);
		if (pending != null) return pending;

		const promise = requests
			.get<SchemasResponse>(`schemas/${dbId}`)
			.then((res): ApiResponse<Schema[]> => {
				if (res.error) return res as unknown as ApiResponse<Schema[]>;
				const raw = res as unknown as SchemasResponse;
				const schemas: Schema[] = raw.schemas.map((s) => ({ ...s, tables: [] }));
				return { data: schemas, count: raw.schemas_count };
			})
			.finally(() => {
				schemasByDbMap.delete(dbId);
			});

		schemasByDbMap.set(dbId, promise);
		return promise;
	},

	/** Tables for one schema; parallel callers with the same key share one HTTP request. */
	getTablesForSchema: (
		schemaId: string,
		opts: { databaseName?: string } = {},
	): Promise<ApiResponse<Table[]>> => {
		const key = [schemaId, opts.databaseName ?? ''].join('\0');
		const pending = tablesBySchemaMap.get(key);
		if (pending != null) return pending;

		const params: Params = {};
		if (opts.databaseName != null && opts.databaseName !== '') {
			params.database_name = opts.databaseName;
		}

		const promise = requests
			.get<ResponseWithCount<Omit<Table, 'columns'>[]>>(`tables/${schemaId}`, params)
			.then((res): ApiResponse<Table[]> => {
				if (res.error) return res as unknown as ApiResponse<Table[]>;
				const tables: Table[] = res.data.map((t) => ({ ...t, columns: [] }));
				return { data: tables, count: tables.length };
			})
			.finally(() => {
				tablesBySchemaMap.delete(key);
			});

		tablesBySchemaMap.set(key, promise);
		return promise;
	},

	/**
	 * Columns for one table, or one page of them when a window is given;
	 * parallel callers reading the same window share one HTTP request.
	 * `total` is what the table has in full, which a pager compares against.
	 */
	getColumnsForTable: (
		tableId: string,
		params: PageParams = {},
	): Promise<ApiPagedResponse<Column[]>> => {
		const key = [tableId, params.skip ?? 0, params.limit ?? ''].join('\0');
		const pending = columnsByTableMap.get(key);
		if (pending != null) return pending;

		const promise = requests
			.get<ColumnsPageResult>(`columns/${tableId}`, pageQuery(params))
			.then((res): ApiPagedResponse<Column[]> => {
				if (res.error) return res as unknown as ApiPagedResponse<Column[]>;
				const envelope = res.data as unknown as ColumnsEnvelope;
				const columns: Column[] = (envelope.columns ?? []).map((c) => ({
					...c,
					database_name: envelope.database_name,
					schema_name: envelope.schema_name,
					table_name: envelope.table_name,
				}));
				return { data: columns, count: columns.length, total: res.total ?? 0 };
			})
			.finally(() => {
				columnsByTableMap.delete(key);
			});

		columnsByTableMap.set(key, promise);
		return promise;
	},

	/** Update mutable properties of any catalog node (Database, Schema, Table, or Column). */
	updateNode: (nodeId: string, patch: NodePatch) =>
		requests.patch<NodeUpdateResponse>(`nodes/${nodeId}`, patch),
};
