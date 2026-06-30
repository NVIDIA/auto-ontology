// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { authClient } from '@/auth/auth-client';
import { Role } from '@/enums/auth';

/** Session shape inferred from the configured Better Auth client. */
export type AppSession = typeof authClient.$Infer.Session;

/** Authenticated user, including the admin plugin's `role` field. */
type AppUser = AppSession['user'];

export type User = Pick<AppUser, 'id' | 'name' | 'email'> & {
	role: Role;
};
