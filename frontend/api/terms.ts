// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { requests } from './requests';
import type { Term, TermAttribute, TermDetail } from '@/types/terms';
import type { ResponseWithError } from './types';

type ListResult = { data: Term[]; count: number };
type ListResponse = ResponseWithError<ListResult>;

type AttributeListResult = { data: TermAttribute[]; count: number };
type AttributeListResponse = ResponseWithError<AttributeListResult>;

type SingleResult = { data: TermDetail };
type SingleResponse = ResponseWithError<SingleResult>;

export const termsApi = {
	list: (): Promise<ListResponse> => requests.get<ListResult>('terms'),
	listColumnAttributes: (): Promise<AttributeListResponse> =>
		requests.get<AttributeListResult>('terms/column-attributes'),
	get: (id: string): Promise<SingleResponse> => requests.get<SingleResult>(`terms/${id}`),
	getColumnAttributes: (id: string): Promise<AttributeListResponse> =>
		requests.get<AttributeListResult>(`terms/${id}/column-attributes`),
};
