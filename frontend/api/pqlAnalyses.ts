// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { requests } from './requests';
import type { PqlAnalysis } from '@/types/analysis';
import type { ResponseWithCount, ResponseWithError } from './types';

type ListResponse = ResponseWithError<ResponseWithCount<PqlAnalysis[]>>;

export type PqlAnalysisCreatePayload = {
	name: string;
	description: string;
	pql: string;
};

type CreateResponse = ResponseWithError<{ data: PqlAnalysis }>;
type UpdateResponse = ResponseWithError<{ data: PqlAnalysis }>;
type DeleteResponse = ResponseWithError<{ data: { id: string } }>;

export const pqlAnalyses = {
	list: (): Promise<ListResponse> =>
		requests.get<ResponseWithCount<PqlAnalysis[]>>('pql-analyses'),
	create: (payload: PqlAnalysisCreatePayload): Promise<CreateResponse> =>
		requests.post<{ data: PqlAnalysis }>('pql-analyses', payload),
	update: (id: string, payload: PqlAnalysisCreatePayload): Promise<UpdateResponse> =>
		requests.put<{ data: PqlAnalysis }>(`pql-analyses/${encodeURIComponent(id)}`, payload),
	delete: (id: string): Promise<DeleteResponse> =>
		requests.delete<{ data: { id: string } }>(`pql-analyses/${encodeURIComponent(id)}`),
};
