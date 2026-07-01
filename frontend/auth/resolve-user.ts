// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Resolve the GSF user behind a request, for handlers that accept both browser
// and service callers.
//
// Browser callers have a Better Auth session. Bearer callers (e.g. AI-Q)
// authenticate with the same NVIDIA SSO identity but have no GSF session, so
// they are mapped to the GSF user that shares the token's email (the account
// they sign into GSF with), falling back to the SSO account link by subject.

import { headers } from 'next/headers';
import { getCurrentSession } from '@/auth/auth-guards';
import { verifyBearer } from '@/auth/bearer';
import { getPrisma } from '@/lib/prisma';

export type ResolvedUser = { id: string; role: string | null };

/**
 * Resolve the GSF user behind the current request — from the session cookie, or
 * from a verified SSO bearer token (matched by email, then by SSO account
 * subject). Returns null when the caller is unauthenticated or has no matching
 * GSF user. This is the single entry point both browser and service callers go
 * through, so every protected API accepts either credential.
 */
export async function resolveUser(): Promise<ResolvedUser | null> {
	const session = await getCurrentSession();
	if (session?.user.id) return { id: session.user.id, role: session.user.role ?? null };

	const principal = await verifyBearer(await headers());
	if (!principal) return null;

	const prisma = getPrisma();

	if (principal.email) {
		const user = await prisma.user.findFirst({
			where: { email: principal.email },
			select: { id: true, role: true },
		});
		if (user) return { id: user.id, role: user.role };
	}

	if (principal.subject) {
		const account = await prisma.account.findFirst({
			where: { accountId: principal.subject },
			select: { user: { select: { id: true, role: true } } },
		});
		if (account?.user) return { id: account.user.id, role: account.user.role };
	}

	return null;
}

/**
 * Convenience wrapper returning just the resolved user id (or null). See
 * {@link resolveUser}.
 */
export async function resolveUserId(): Promise<string | null> {
	return (await resolveUser())?.id ?? null;
}
