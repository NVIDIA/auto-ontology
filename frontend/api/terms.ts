// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { requests } from './requests';
import type {
	ColumnAttribute,
	RelatedTerm,
	RelatedTermCount,
	Term,
	TermDetail,
} from '@/types/terms';
import type { ResponseWithError } from './types';

type ListResult = { data: Term[]; count: number };
type ListResponse = ResponseWithError<ListResult>;

type AttributeListResult = { data: ColumnAttribute[]; count: number };
type AttributeListResponse = ResponseWithError<AttributeListResult>;

type SingleResult = { data: TermDetail };
type SingleResponse = ResponseWithError<SingleResult>;

type RelatedTermsResult = { data: RelatedTerm[]; count: number };
type RelatedTermsResponse = ResponseWithError<RelatedTermsResult>;

type RelatedCountsResult = { data: RelatedTermCount[]; count: number };
type RelatedCountsResponse = ResponseWithError<RelatedCountsResult>;

export const termsApi = {
	list: (): Promise<ListResponse> => requests.get<ListResult>('terms'),
	listColumnAttributes: (): Promise<AttributeListResponse> =>
		requests.get<AttributeListResult>('terms/column-attributes'),
	listRelatedCounts: (): Promise<RelatedCountsResponse> =>
		requests.get<RelatedCountsResult>('terms/related-counts'),
	get: (id: string): Promise<SingleResponse> => requests.get<SingleResult>(`terms/${id}`),
	getColumnAttributes: (id: string): Promise<AttributeListResponse> =>
		requests.get<AttributeListResult>(`terms/${id}/column-attributes`),
	getRelatedTerms: (id: string): Promise<RelatedTermsResponse> =>
		requests.get<RelatedTermsResult>(`terms/${id}/related-terms`),
};
