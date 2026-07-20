// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { requests } from './requests';
import type {
	ColumnAttribute,
	RelatedTerm,
	RelatedTermCount,
	SqlAttribute,
	Term,
	TermCount,
	TermDetail,
} from '@/types/terms';
import type { ResponseWithError } from './types';

type ListResult = { data: Term[]; count: number };
type ListResponse = ResponseWithError<ListResult>;

type AttributeListResult = { data: ColumnAttribute[]; count: number };
type AttributeListResponse = ResponseWithError<AttributeListResult>;

type SqlAttributeListResult = { data: SqlAttribute[]; count: number };
type SqlAttributeListResponse = ResponseWithError<SqlAttributeListResult>;

type SingleResult = { data: TermDetail };
type SingleResponse = ResponseWithError<SingleResult>;
type UpdateResult = { data: { id: string; name: string; description: string | null } };
type UpdateResponse = ResponseWithError<UpdateResult>;
type TermUpdatePayload = { name?: string; description?: string | null };

type RelatedTermsResult = { data: RelatedTerm[]; count: number };
type RelatedTermsResponse = ResponseWithError<RelatedTermsResult>;

type RelatedCountsResult = { data: RelatedTermCount[]; count: number };
type RelatedCountsResponse = ResponseWithError<RelatedCountsResult>;

type AttributeCountsResult = { data: TermCount[]; count: number };
type AttributeCountsResponse = ResponseWithError<AttributeCountsResult>;

export type TermsListParams = {
	/** Case-insensitive substring filter on the term name. */
	q?: string;
};

export const termsApi = {
	list: (params?: TermsListParams): Promise<ListResponse> =>
		requests.get<ListResult>('terms', params?.q ? { q: params.q } : {}),
	listColumnAttributes: (): Promise<AttributeListResponse> =>
		requests.get<AttributeListResult>('terms/column-attributes'),
	listSqlAttributes: (): Promise<SqlAttributeListResponse> =>
		requests.get<SqlAttributeListResult>('terms/sql-attributes'),
	listRelatedCounts: (): Promise<RelatedCountsResponse> =>
		requests.get<RelatedCountsResult>('terms/related-counts'),
	listColumnAttributeCounts: (): Promise<AttributeCountsResponse> =>
		requests.get<AttributeCountsResult>('terms/column-attributes/counts'),
	listSqlAttributeCounts: (): Promise<AttributeCountsResponse> =>
		requests.get<AttributeCountsResult>('terms/sql-attributes/counts'),
	get: (id: string): Promise<SingleResponse> => requests.get<SingleResult>(`terms/${id}`),
	update: (id: string, payload: TermUpdatePayload): Promise<UpdateResponse> =>
		requests.patch<UpdateResult>(`terms/${encodeURIComponent(id)}`, payload),
	getColumnAttributes: (id: string): Promise<AttributeListResponse> =>
		requests.get<AttributeListResult>(`terms/${id}/column-attributes`),
	getSqlAttributes: (id: string): Promise<SqlAttributeListResponse> =>
		requests.get<SqlAttributeListResult>(`terms/${id}/sql-attributes`),
	getRelatedTerms: (id: string): Promise<RelatedTermsResponse> =>
		requests.get<RelatedTermsResult>(`terms/${id}/related-terms`),
};
