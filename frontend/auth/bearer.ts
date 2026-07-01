// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Bearer-token authentication for service-to-service calls.
//
// GSF normally authenticates browsers via a Better Auth session cookie. But
// AI-Q calls our chat API on behalf of a signed-in user, and both products
// trust the *same* NVIDIA SSO provider. So instead of a cookie, AI-Q forwards
// the user's SSO id token as `Authorization: Bearer <jwt>`, and we verify it
// here against that provider's JWKS — the same issuer an interactive SSO login
// would use. No shared service secret is involved; every call is attributed to
// the token's subject.
//
// We deliberately verify only signature + issuer + expiry, NOT audience: the
// token was minted for AI-Q's OAuth client, so its `aud` is AI-Q's client id,
// not ours. Trusting "any token our SSO provider signed" is the intended model.

import { createRemoteJWKSet, decodeJwt, jwtVerify, type JWTPayload } from 'jose';
import { getPrisma } from '@/lib/prisma';

export type BearerPrincipal = {
	/** OIDC subject claim (stable per-user id from the IdP). */
	subject: string;
	email?: string;
	name?: string;
};

type ProviderJwks = {
	issuer: string;
	jwks: ReturnType<typeof createRemoteJWKSet>;
};

// Cache the remote JWKS per provider so we don't refetch keys on every request;
// createRemoteJWKSet itself caches and rotates keys internally. Keyed by the
// JWKS URL so a reconfigured provider transparently gets a fresh set.
const jwksCache = new Map<string, ProviderJwks>();

const extractBearer = (headers: Headers): string | null => {
	const header = headers.get('authorization');
	if (!header) return null;
	const [scheme, token] = header.split(' ');
	if (scheme?.toLowerCase() !== 'bearer' || !token) return null;
	return token.trim() || null;
};

/**
 * Resolve the configured SSO provider's issuer + JWKS URL. Returns null when no
 * provider is configured or it lacks a JWKS endpoint (e.g. SAML-only).
 */
const resolveProviderJwks = async (): Promise<ProviderJwks | null> => {
	const provider = await getPrisma().ssoProvider.findFirst();
	if (!provider?.oidcConfig) {
		console.warn('[bearer] no SSO provider / oidcConfig configured');
		return null;
	}

	let config: Record<string, unknown>;
	try {
		config = JSON.parse(provider.oidcConfig) as Record<string, unknown>;
	} catch {
		console.warn('[bearer] oidcConfig is not valid JSON');
		return null;
	}

	// Better Auth may store the discovery URL under a few different keys
	// depending on version; accept the common spellings.
	const jwksUrl =
		(config.jwksEndpoint as string | undefined) ??
		(config.jwksUrl as string | undefined) ??
		(config.jwks_uri as string | undefined);
	const issuer = provider.issuer ?? (config.issuer as string | undefined) ?? undefined;

	if (!jwksUrl || !issuer) {
		console.warn(
			`[bearer] provider missing jwksUrl or issuer (jwksUrl=${jwksUrl ?? 'none'}, issuer=${issuer ?? 'none'}, oidcConfig keys=${Object.keys(config).join(',')})`,
		);
		return null;
	}

	const cached = jwksCache.get(jwksUrl);
	if (cached && cached.issuer === issuer) return cached;

	const entry: ProviderJwks = { issuer, jwks: createRemoteJWKSet(new URL(jwksUrl)) };
	jwksCache.set(jwksUrl, entry);
	return entry;
};

/**
 * Verify a `Authorization: Bearer <jwt>` header against the configured SSO
 * provider. Returns the verified principal, or null when there is no bearer
 * token, no usable provider, or the token fails verification.
 */
export async function verifyBearer(headers: Headers): Promise<BearerPrincipal | null> {
	const token = extractBearer(headers);
	if (!token) return null;

	const provider = await resolveProviderJwks();
	if (!provider) return null;

	let payload: JWTPayload;
	try {
		({ payload } = await jwtVerify(token, provider.jwks, { issuer: provider.issuer }));
	} catch (err) {
		// Decode (without verifying) to surface why the token was rejected —
		// issuer mismatch and expiry are the usual culprits for a cross-service
		// SSO token.
		let claims = '';
		try {
			const c = decodeJwt(token);
			claims = ` token.iss=${c.iss} token.aud=${JSON.stringify(c.aud)} token.exp=${c.exp} expected.iss=${provider.issuer}`;
		} catch {
			claims = ' (token is not a decodable JWT)';
		}
		console.warn(
			`[bearer] jwtVerify failed: ${err instanceof Error ? err.message : String(err)}.${claims}`,
		);
		return null;
	}

	if (!payload.sub) return null;
	console.warn(`[bearer] verified OK sub=${payload.sub} iss=${payload.iss}`);
	return {
		subject: payload.sub,
		email: typeof payload.email === 'string' ? payload.email : undefined,
		name: typeof payload.name === 'string' ? payload.name : undefined,
	};
}
