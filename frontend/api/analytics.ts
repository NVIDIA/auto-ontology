// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { requests } from './requests';
import type { ConversationAnalytics } from '@/types/analytics';
import type { ResponseWithError } from './types';

type ListResult = { data: ConversationAnalytics[]; total: number };
type ListResponse = ResponseWithError<ListResult>;
type RowResponse = ResponseWithError<ConversationAnalytics>;

type UpdatePayload = {
	responseTimestamp: number;
	response: string;
	sql?: string;
	responseMessageId?: string;
};

export const analyticsApi = {
	list: (params: { skip?: number; limit?: number } = {}): Promise<ListResponse> =>
		requests.get<ListResult>('analytics', params),

	create: (questionMessageId: string, question: string): Promise<RowResponse> =>
		requests.post<ConversationAnalytics>('analytics', { questionMessageId, question }),

	update: (analyticsId: string, payload: UpdatePayload): Promise<RowResponse> =>
		requests.patch<ConversationAnalytics>(
			`analytics/${encodeURIComponent(analyticsId)}`,
			payload,
		),
};
