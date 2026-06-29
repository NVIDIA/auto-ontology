// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { requests } from './requests';
import type { ConversationAnalytics } from '@/types/analytics';
import type { ResponseWithError } from './types';

type ListResult = { data: ConversationAnalytics[]; total: number };
type ListResponse = ResponseWithError<ListResult>;

// Analytics rows are written server-side in the chat proxy route; the client
// only reads the report.
export const analyticsApi = {
	list: (params: { skip?: number; limit?: number } = {}): Promise<ListResponse> =>
		requests.get<ListResult>('analytics', params),
};
