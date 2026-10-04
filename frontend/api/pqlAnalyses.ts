// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { pageQuery, requests } from './requests';
import type { PqlAnalysis } from '@/types/analysis';
import type { ApiPagedResponse, PageParams, ResponseWithCount, ResponseWithError } from './types';

/** One page of the list: `count` is this page, `total` every match. */
type ListResult = ResponseWithCount<PqlAnalysis[]> & { total: number };
type ListResponse = ApiPagedResponse<PqlAnalysis[]>;

export type PqlAnalysesListParams = PageParams & {
	/** Case-insensitive substring filter on the analysis name. */
	query?: string;
};

export type PqlAnalysisCreatePayload = {
	name: string;
	description: string;
	pql: string;
};

type CreateResponse = ResponseWithError<{ data: PqlAnalysis }>;
type UpdateResponse = ResponseWithError<{ data: PqlAnalysis }>;
type DeleteResponse = ResponseWithError<{ data: { id: string } }>;

export const pqlAnalyses = {
	list: (params?: PqlAnalysesListParams): Promise<ListResponse> =>
		requests.get<ListResult>('pql-analyses', {
			...(params?.query ? { query: params.query } : {}),
			...pageQuery(params),
		}),
	create: (payload: PqlAnalysisCreatePayload): Promise<CreateResponse> =>
		requests.post<{ data: PqlAnalysis }>('pql-analyses', payload),
	update: (id: string, payload: PqlAnalysisCreatePayload): Promise<UpdateResponse> =>
		requests.put<{ data: PqlAnalysis }>(`pql-analyses/${encodeURIComponent(id)}`, payload),
	delete: (id: string): Promise<DeleteResponse> =>
		requests.delete<{ data: { id: string } }>(`pql-analyses/${encodeURIComponent(id)}`),
};
