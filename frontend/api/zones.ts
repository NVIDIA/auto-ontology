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
	ZoneUser,
} from '@/types/zones';

export const zonesApi = {
	getAll: (userId: string): Promise<ApiResponse<Zone[]>> =>
		requests.get('zones', { uid: userId }),

	getById: (zoneId: string, userId: string): Promise<ApiResponse<ZoneDetail>> =>
		requests.get(`zones/${zoneId}`, { uid: userId }),

	create: (input: ZoneCreateInput): Promise<ResponseWithError<{ data: ZoneCreated }>> =>
		requests.post('zones', input),

	update: (
		zoneId: string,
		input: ZoneUpdateInput,
	): Promise<ResponseWithError<{ data: ZoneDetail }>> => requests.patch(`zones/${zoneId}`, input),

	delete: (zoneId: string): Promise<ResponseWithError<{ data: { id: string } }>> =>
		requests.delete(`zones/${zoneId}`),

	listAccess: (
		zoneId: string,
		adminId: string,
	): Promise<ResponseWithError<{ data: ZoneUser[]; count: number }>> =>
		requests.get(`zones/${zoneId}/access`, { admin_uid: adminId }),

	grantAccess: (
		zoneId: string,
		userId: string,
		adminId: string,
	): Promise<ResponseWithError<{ data: { zone_id: string; user_id: string } }>> =>
		requests.post(`zones/${zoneId}/access?admin_uid=${encodeURIComponent(adminId)}`, {
			user_id: userId,
		}),

	revokeAccess: (
		zoneId: string,
		userId: string,
		adminId: string,
	): Promise<ResponseWithError<{ data: { zone_id: string; user_id: string } }>> =>
		requests.delete(
			`zones/${zoneId}/access/${encodeURIComponent(userId)}?admin_uid=${encodeURIComponent(adminId)}`,
		),
};
