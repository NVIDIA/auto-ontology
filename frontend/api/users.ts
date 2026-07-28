// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { authClient } from '@/auth/auth-client';
import { requests } from './requests';
import { Role } from '@/enums/auth';
import type { User } from '@/types/auth';
import type { ResponseWithError } from './types';

type ListResult = { users: User[]; error: string | null };

export const usersApi = {
	list: async (): Promise<ListResult> => {
		const result = await authClient.admin.listUsers({ query: { limit: 200 } });
		if (result.error) {
			return { users: [], error: result.error.message ?? 'Failed to load users.' };
		}
		const users: User[] = (result.data?.users ?? []).map((user) => ({
			id: user.id,
			name: user.name,
			email: user.email,
			role: (user.role as Role) ?? Role.Viewer,
		}));
		return { users, error: null };
	},

	setRole: (userId: string, role: Role) => authClient.admin.setRole({ userId, role }),

	remove: (userId: string) => authClient.admin.removeUser({ userId }),

	// Current user's "Visualize SQL Results" preference (Settings > Agent Settings).
	getVisualization: (): Promise<ResponseWithError<{ visualization: boolean }>> =>
		requests.get('users/visualization'),

	setVisualization: (
		visualization: boolean,
	): Promise<ResponseWithError<{ visualization: boolean }>> =>
		requests.put('users/visualization', { visualization }),
};
