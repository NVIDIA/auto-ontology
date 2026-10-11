// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { requests } from './requests';
import type { ApiResponse, ResponseWithError } from './types';
import type { Group, GroupWriteInput } from '@/types/groups';

export const groupsApi = {
	/** Every group, ordered by name, with its members. */
	getAll: (): Promise<ApiResponse<Group[]>> => requests.get('groups'),

	/** 409 when the name is taken — `message` is the backend's own wording. */
	create: (input: GroupWriteInput): Promise<ResponseWithError<{ data: Group }>> =>
		requests.post('groups', input),

	/**
	 * Replace the fields that are sent. `user_ids` replaces membership; leaving
	 * it out leaves the members alone.
	 */
	update: (
		groupId: string,
		input: GroupWriteInput,
	): Promise<ResponseWithError<{ data: Group }>> => requests.patch(`groups/${groupId}`, input),

	delete: (groupId: string): Promise<ResponseWithError<{ data: { id: string } }>> =>
		requests.delete(`groups/${groupId}`),
};
