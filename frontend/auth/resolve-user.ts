// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Resolve the GSF user behind a request, for handlers that accept both browser
// and service callers.
//
// Three credentials are accepted, in order:
//   1. A Better Auth session cookie (browser).
//   2. A GSF API token, `x-api-key: gsf_...` (scripts, CI — see auth/api-token.ts).
//      Checked before the SSO bearer because both may arrive as `Authorization:
//      Bearer`, and only the API-token path recognises the `gsf_` prefix.
//   3. An SSO id token (e.g. AI-Q), which authenticates with the same NVIDIA SSO
//      identity but has no GSF session, so it is mapped to the GSF user sharing
//      the token's email, falling back to the SSO account link by subject.

import { headers } from 'next/headers';
import { getCurrentSession } from '@/auth/auth-guards';
import { extractApiToken, verifyApiToken } from '@/auth/api-token';
import { verifyBearer } from '@/auth/bearer';
import { getPrisma } from '@/lib/prisma';

export type ResolvedUser = { id: string; role: string | null };

/**
 * Resolve the GSF user behind the current request — from the session cookie, a
 * GSF API token, or a verified SSO bearer token (matched by email, then by SSO
 * account subject). Returns null when the caller is unauthenticated or has no
 * matching GSF user. This is the single entry point browser, script, and service
 * callers all go through, so every protected API accepts any of the three
 * credentials.
 */
export async function resolveUser(): Promise<ResolvedUser | null> {
	const session = await getCurrentSession();
	if (session?.user.id) return { id: session.user.id, role: session.user.role ?? null };

	const requestHeaders = await headers();

	// A GSF token is unambiguous, so presenting one settles which credential is
	// being offered: a revoked or expired token is a 401, not an invitation to
	// try it again as an SSO id token (which would only log a confusing "not a
	// decodable JWT" on the way to the same answer).
	if (extractApiToken(requestHeaders)) return verifyApiToken(requestHeaders);

	const principal = await verifyBearer(requestHeaders);
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
