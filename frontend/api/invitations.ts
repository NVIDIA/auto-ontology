// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { requests } from '@/api/requests';
import { Role } from '@/enums/auth';
import type { Invitation } from '@/types/invitation';

export type CreatedInvitation = {
	url: string;
	email: string;
	expires_at: string;
};

export const invitationsApi = {
	list: () => requests.get<{ invitations: Invitation[] }>('invitations'),

	create: (input: { email: string; name?: string; role: Role }) =>
		requests.post<CreatedInvitation>('invitations', input),

	resetPassword: (userId: string) =>
		requests.post<CreatedInvitation>('invitations/reset', { userId }),

	accept: (token: string, password: string) =>
		requests.post<{ email: string }>(`invitations/${encodeURIComponent(token)}/accept`, {
			password,
		}),

	remove: (id: string) =>
		requests.delete<{ id: string }>(`invitations/${encodeURIComponent(id)}`),
};
