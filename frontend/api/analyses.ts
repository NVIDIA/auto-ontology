// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { pageQuery, requests } from './requests';
import type { CustomAnalysis } from '@/types/analysis';
import type { ApiPagedResponse, PageParams, ResponseWithCount, ResponseWithError } from './types';

/** One page of the list: `count` is this page, `total` every match. */
type ListResult = ResponseWithCount<CustomAnalysis[]> & { total: number };
type ListResponse = ApiPagedResponse<CustomAnalysis[]>;

export type CustomAnalysesListParams = PageParams & {
	/** Case-insensitive substring filter on the analysis name. */
	query?: string;
};

export type CustomAnalysisCreatePayload = {
	name: string;
	description: string;
	sql: string;
};

type CreateResponse = ResponseWithError<{ data: CustomAnalysis }>;
type UpdateResponse = ResponseWithError<{ data: CustomAnalysis }>;
type DeleteResponse = ResponseWithError<{ data: { id: string } }>;
type ValidateResponse = ResponseWithError<{ data: { valid: boolean; sql: string } }>;

export const analyses = {
	list: (params?: CustomAnalysesListParams): Promise<ListResponse> =>
		requests.get<ListResult>('custom-analyses', {
			...(params?.query ? { query: params.query } : {}),
			...pageQuery(params),
		}),
	validate: (sql: string): Promise<ValidateResponse> =>
		requests.post<{ data: { valid: boolean; sql: string } }>('custom-analyses/validate', {
			sql,
		}),
	create: (payload: CustomAnalysisCreatePayload): Promise<CreateResponse> =>
		requests.post<{ data: CustomAnalysis }>('custom-analyses', payload),
	update: (id: string, payload: CustomAnalysisCreatePayload): Promise<UpdateResponse> =>
		requests.put<{ data: CustomAnalysis }>(
			`custom-analyses/${encodeURIComponent(id)}`,
			payload,
		),
	delete: (id: string): Promise<DeleteResponse> =>
		requests.delete<{ data: { id: string } }>(`custom-analyses/${encodeURIComponent(id)}`),
};
