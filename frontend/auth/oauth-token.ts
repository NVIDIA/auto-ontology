// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// OAuth access-token authentication, for callers that signed in through Auto Ontology.
//
// Auto Ontology is itself the authorization server an MCP client logs in against (see the
// `mcp` plugin in auth/auth.ts). A client sends the user through Auto Ontology's normal
// login page, receives an access token, and presents it here. So unlike the SSO
// bearer path — which trusts a token another product's IdP signed — these tokens
// are ours, and validating one is a lookup in our own database.
//
// The token carries no permissions of its own: it resolves to the user who
// approved it, and the request is then authorized against that user's role, the
// same way a browser or API-token request is.

import { auth } from '@/auth/auth';
import { getPrisma } from '@/lib/prisma';

// A Auto Ontology access token is opaque — random characters, no structure. An SSO id
// token is a JWT: three dot-separated segments. Telling them apart by shape
// lets each credential go straight to the code that can verify it, instead of
// failing JWT verification first and logging a misleading reason.
const JWT_SHAPE = /^[\w-]+\.[\w-]+\.[\w-]+$/;

/**
 * Pull an opaque `Authorization: Bearer` token out of the request headers —
 * i.e. one that is not a JWT.
 *
 * Callers must check for a Auto Ontology API token first (see auth/api-token.ts); those
 * are opaque too, so this would otherwise claim them.
 */
export function extractOAuthToken(headers: Headers): string | null {
	const authorization = headers.get('authorization');
	if (!authorization) return null;

	const [scheme, value] = authorization.split(' ');
	if (scheme?.toLowerCase() !== 'bearer') return null;

	const token = value?.trim();
	if (!token || JWT_SHAPE.test(token)) return null;
	return token;
}

/**
 * Verify a Auto Ontology-issued OAuth access token and resolve the user who granted it.
 *
 * Returns null when the token is unknown, expired, or its owner no longer
 * exists or has been banned.
 */
export async function verifyOAuthToken(
	headers: Headers,
): Promise<{ id: string; role: string | null } | null> {
	// UserInfo accepts the bearer directly and validates opaque tokens against
	// the stored grant, including expiry, revocation, client, and session state.
	const response = await auth.api.oauth2UserInfo({ headers, asResponse: true });
	if (!response.ok) {
		if (response.status >= 500) {
			throw new Error(`OAuth token validation failed with HTTP ${response.status}`);
		}
		return null;
	}
	const userInfo: unknown = await response.json();
	const userId =
		typeof userInfo === 'object' &&
		userInfo !== null &&
		'sub' in userInfo &&
		typeof userInfo.sub === 'string'
			? userInfo.sub
			: null;
	if (!userId) return null;

	// Re-read the role rather than trusting the grant, so a role change or a ban
	// takes effect immediately instead of at token expiry.
	const user = await getPrisma().user.findUnique({
		where: { id: userId },
		select: { id: true, role: true, banned: true },
	});
	if (!user || user.banned) return null;

	return { id: user.id, role: user.role };
}
