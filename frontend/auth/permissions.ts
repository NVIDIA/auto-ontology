// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { roles, type PermissionRequest } from '@/auth/auth-access';
import { Role } from '@/enums/auth';
import type { ResolvedUser } from '@/auth/resolve-user';

export type { PermissionRequest };

/**
 * Does this user's role grant the requested permission(s)?
 *
 * Checks the role's access-control statements in memory (no DB / network) via
 * the same `authorize()` Better Auth's client `checkRolePermission` uses. We
 * check by the role we already resolved (see `resolveUser`) rather than re-
 * reading it by user id — the role is authoritative for the current request and
 * this avoids a redundant lookup.
 *
 * The default connector is AND: every requested resource/action must be allowed.
 */
export function userCan(user: ResolvedUser, permissions: PermissionRequest): boolean {
	if (!user.role) return false;
	const role = roles[user.role as Role];
	if (!role) return false;
	return role.authorize(permissions).success === true;
}
