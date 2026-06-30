// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { NextResponse } from 'next/server';
import { resolveUser } from '@/auth/resolve-user';
import { Role } from '@/enums/auth';

// Every guard below resolves the caller via resolveUser(), which accepts either
// a Better Auth session cookie (browser) or a verified SSO bearer token (service
// callers such as AI-Q). So all protected APIs transparently accept both.
//
// Route handlers are NOT covered by the page auth gate, and the /api/* middleware
// gate (see proxy.ts) is only an optimistic presence check — a forged or stale
// credential passes it — so these perform the real, server-side validation.

/**
 * Auth guard for API route handlers. Returns a 401 Response to return
 * immediately when the caller can't be resolved, or null when authenticated.
 *
 * Usage:
 *   const denied = await requireApiAuth();
 *   if (denied) return denied;
 */
export async function requireApiAuth(): Promise<NextResponse | null> {
	const user = await resolveUser();
	if (!user) {
		return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
	}
	return null;
}

/**
 * Like {@link requireApiAuth}, but also returns the authenticated user's id for
 * handlers that scope data per user.
 *
 * Usage:
 *   const { userId, deny } = await getApiUser();
 *   if (deny) return deny;
 *   // ...use userId
 */
export async function getApiUser(): Promise<
	{ userId: string; deny: null } | { userId: null; deny: NextResponse }
> {
	const user = await resolveUser();
	if (!user) {
		return {
			userId: null,
			deny: NextResponse.json({ error: 'Unauthorized' }, { status: 401 }),
		};
	}
	return { userId: user.id, deny: null };
}

/**
 * Admin-only guard for API route handlers. Returns 401 when the caller can't be
 * resolved, 403 when resolved but not an admin, or null when the caller is an
 * admin.
 */
export async function requireApiAdmin(): Promise<NextResponse | null> {
	const user = await resolveUser();
	if (!user) {
		return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
	}
	if (user.role !== Role.Admin) {
		return NextResponse.json({ error: 'Forbidden' }, { status: 403 });
	}
	return null;
}
