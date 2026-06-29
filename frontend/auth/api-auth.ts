// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { NextResponse } from 'next/server';
import { getCurrentSession } from '@/auth/auth-guards';
import { Role } from '@/enums/auth';

/**
 * Auth guard for API route handlers.
 *
 * Route handlers are NOT covered by the page auth gate, so each protected
 * handler must verify the session itself. The /api/* middleware gate
 * (see proxy.ts) is only an optimistic cookie-presence check — a forged or
 * stale cookie passes it — so this performs the real, server-side session
 * validation.
 *
 * Returns a 401 Response to return immediately when unauthenticated, or null
 * when the request is authenticated.
 *
 * Usage:
 *   const denied = await requireApiAuth();
 *   if (denied) return denied;
 */
export async function requireApiAuth(): Promise<NextResponse | null> {
	const session = await getCurrentSession();
	if (!session) {
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
	const session = await getCurrentSession();
	if (!session) {
		return {
			userId: null,
			deny: NextResponse.json({ error: 'Unauthorized' }, { status: 401 }),
		};
	}
	return { userId: session.user.id, deny: null };
}

/**
 * Admin-only guard for API route handlers. Returns 401 when unauthenticated,
 * 403 when authenticated but not an admin, or null when the caller is an admin.
 */
export async function requireApiAdmin(): Promise<NextResponse | null> {
	const session = await getCurrentSession();
	if (!session) {
		return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
	}
	if (session.user.role !== Role.Admin) {
		return NextResponse.json({ error: 'Forbidden' }, { status: 403 });
	}
	return null;
}
