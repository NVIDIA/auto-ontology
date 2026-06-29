// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { requests } from './requests';
import type { ApiResponse, ResponseWithError } from './types';
import type {
	Zone,
	ZoneCreated,
	ZoneCreateInput,
	ZoneDetail,
	ZoneUpdateInput,
} from '@/types/zones';

export const zonesApi = {
	getAll: (): Promise<ApiResponse<Zone[]>> => requests.get('zones'),
	getById: (zoneId: string): Promise<ApiResponse<ZoneDetail>> => requests.get(`zones/${zoneId}`),
	create: (input: ZoneCreateInput): Promise<ResponseWithError<{ data: ZoneCreated }>> =>
		requests.post('zones', input),
	update: (
		zoneId: string,
		input: ZoneUpdateInput,
	): Promise<ResponseWithError<{ data: ZoneDetail }>> => requests.patch(`zones/${zoneId}`, input),
	delete: (zoneId: string): Promise<ResponseWithError<{ data: { id: string } }>> =>
		requests.delete(`zones/${zoneId}`),
};
