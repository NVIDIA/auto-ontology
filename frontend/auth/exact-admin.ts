// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Role } from '@/enums/auth';

/**
 * The role string Better Auth persists. A list is joined with commas, so
 * `["admin", "viewer"]` is stored as `admin,viewer`.
 */
export const storedRole = (roles: readonly string[]): string => roles.join(',');

/**
 * True only for the exact `admin` value. This is the same predicate as
 * `is_admin` and the users table; a comma-joined role is not an admin.
 */
export const isExactAdmin = (role: string | null | undefined): boolean => role === Role.Admin;

/**
 * The target is the last exact admin, and this change would leave none.
 * `nextRole` is null when the account is deleted.
 */
export const removesLastExactAdmin = (
	targetRole: string | null | undefined,
	nextRole: string | null,
	otherExactAdminCount: number,
): boolean => isExactAdmin(targetRole) && !isExactAdmin(nextRole) && otherExactAdminCount === 0;
