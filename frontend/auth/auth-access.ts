// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { createAccessControl } from 'better-auth/plugins/access';
import { adminAc, defaultStatements } from 'better-auth/plugins/admin/access';
import { Role } from '@/enums/auth';

/**
 * Access-control roles shared by the Better Auth server and client so that
 * `admin` and `viewer` are the only recognized roles (replacing the built-in
 * `admin`/`user` pair). Admins keep full user-management permissions; viewers
 * get none.
 */
const statement = { ...defaultStatements } as const;

export const ac = createAccessControl(statement);

export const roles = {
	[Role.Admin]: ac.newRole({ ...adminAc.statements }),
	[Role.Viewer]: ac.newRole({}),
};
