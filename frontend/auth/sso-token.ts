// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Resolve the caller's SSO token so the backend can exchange it for a Databricks
// access token and run their queries under their own Unity Catalog grants.
//
// Whether that exchange actually happens is a per-connection setting the backend
// owns ("Authenticate as signed-in user"). We just supply the token when one is
// available; the backend decides whether it is required.
//
// Two kinds of caller reach the chat API:
//   * Service callers (e.g. AI-Q) already present `Authorization: Bearer <jwt>` —
//     that token is forwarded verbatim.
//   * Browser users authenticate with a Better Auth session cookie and carry no
//     bearer token, so we read the JWT Better Auth stored on their SSO account
//     link at sign-in.
//
// Id tokens are short-lived (an hour at NVIDIA's IdP) and Better Auth only writes
// them at sign-in, so the stored copy goes stale mid-session and Databricks
// rejects it with `invalid_grant`. We therefore check `exp` before using one and
// refresh it against the provider's token endpoint when it has aged out.
//
// Password (`credential`) accounts have no SSO token by construction, so an
// admin signed in with email+password gets none — with the feature enabled the
// chat request is refused rather than run under the connection's stored PAT.

import { getPrisma } from '@/lib/prisma';

/** Better Auth's providerId for local email+password accounts (never an SSO link). */
const CREDENTIAL_PROVIDER = 'credential';

// Treat a token expiring within this window as already stale, so it can't lapse
// between our check and Databricks validating it.
const EXPIRY_SKEW_MS = 60_000;

type OidcConfig = {
	tokenEndpoint?: string;
	token_endpoint?: string;
	clientId?: string;
	clientSecret?: string;
};

const extractBearer = (headers: Headers): string | null => {
	const header = headers.get('authorization');
	if (!header) return null;
	const [scheme, token] = header.split(' ');
	if (scheme?.toLowerCase() !== 'bearer' || !token) return null;
	return token.trim() || null;
};

/**
 * Read a JWT's `exp` claim, in milliseconds. Returns null when the token isn't a
 * decodable JWT — callers then fall back to the stored expiry column.
 *
 * The signature is deliberately not verified: this is our own staleness check,
 * not an authorisation decision. Databricks verifies the token for real.
 */
const jwtExpiryMs = (token: string): number | null => {
	const payload = token.split('.')[1];
	if (!payload) return null;
	try {
		const json = Buffer.from(payload, 'base64url').toString('utf8');
		const { exp } = JSON.parse(json) as { exp?: unknown };
		return typeof exp === 'number' ? exp * 1000 : null;
	} catch {
		return null;
	}
};

const isFresh = (expiryMs: number | null): boolean =>
	expiryMs != null && expiryMs - EXPIRY_SKEW_MS > Date.now();

/** The configured SSO provider's token endpoint and client credentials. */
const resolveTokenEndpoint = async (): Promise<{
	url: string;
	clientId: string;
	clientSecret: string;
} | null> => {
	const provider = await getPrisma().ssoProvider.findFirst({ select: { oidcConfig: true } });
	if (!provider?.oidcConfig) return null;

	let config: OidcConfig;
	try {
		config = JSON.parse(provider.oidcConfig) as OidcConfig;
	} catch {
		console.warn('[sso-token] oidcConfig is not valid JSON');
		return null;
	}

	const url = config.tokenEndpoint ?? config.token_endpoint;
	if (!url || !config.clientId || !config.clientSecret) {
		console.warn('[sso-token] provider is missing tokenEndpoint or client credentials');
		return null;
	}
	return { url, clientId: config.clientId, clientSecret: config.clientSecret };
};

/**
 * Exchange the account's refresh token for a fresh id token and persist it.
 *
 * Returns the new id token, or null when the provider has no usable config, no
 * refresh token is stored, or the refresh is rejected (a revoked or expired
 * refresh token means the user must sign in again).
 */
const refreshIdToken = async (accountId: string, refreshToken: string): Promise<string | null> => {
	const endpoint = await resolveTokenEndpoint();
	if (endpoint == null) return null;

	let response: Response;
	try {
		response = await fetch(endpoint.url, {
			method: 'POST',
			headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
			body: new URLSearchParams({
				grant_type: 'refresh_token',
				refresh_token: refreshToken,
				client_id: endpoint.clientId,
				client_secret: endpoint.clientSecret,
			}),
		});
	} catch (error) {
		console.warn('[sso-token] refresh request failed', error);
		return null;
	}

	if (!response.ok) {
		// The body can echo the refresh token back, so only the status is logged.
		console.warn(`[sso-token] refresh rejected with HTTP ${response.status}`);
		return null;
	}

	let payload: {
		id_token?: string;
		access_token?: string;
		refresh_token?: string;
		expires_in?: number;
	};
	try {
		payload = (await response.json()) as typeof payload;
	} catch {
		console.warn('[sso-token] refresh returned a malformed body');
		return null;
	}

	if (!payload.id_token) {
		console.warn('[sso-token] refresh returned no id_token');
		return null;
	}

	// Persist the new tokens. If the write fails (e.g. transient DB blip) we
	// still return the fresh id_token so this request succeeds. On the next
	// refresh the stored (now-invalid, rotated) refresh_token will fail and the
	// user will be asked to re-authenticate — that is the correct degradation.
	try {
		await getPrisma().account.update({
			where: { id: accountId },
			data: {
				idToken: payload.id_token,
				...(payload.access_token != null ? { accessToken: payload.access_token } : {}),
				// Providers may rotate the refresh token; keep the old one when they don't.
				...(payload.refresh_token != null ? { refreshToken: payload.refresh_token } : {}),
				...(payload.expires_in != null
					? { accessTokenExpiresAt: new Date(Date.now() + payload.expires_in * 1000) }
					: {}),
			},
		});
	} catch (error) {
		console.warn(
			'[sso-token] failed to persist refreshed tokens — user may need to re-authenticate after next expiry',
			error,
		);
	}

	return payload.id_token;
};

/**
 * Return a currently-valid SSO JWT for the request, or null when none can be
 * obtained.
 *
 * Prefers a forwarded bearer token, then the account's id token, refreshing it
 * when stale, then the account's access token while it is still valid.
 */
export async function resolveSubjectToken(
	headers: Headers,
	userId: string,
): Promise<string | null> {
	const forwarded = extractBearer(headers);
	if (forwarded) return forwarded;

	const account = await getPrisma().account.findFirst({
		where: { userId, providerId: { not: CREDENTIAL_PROVIDER } },
		select: {
			id: true,
			idToken: true,
			accessToken: true,
			accessTokenExpiresAt: true,
			refreshToken: true,
		},
		orderBy: { updatedAt: 'desc' },
	});
	if (!account) return null;

	if (account.idToken && isFresh(jwtExpiryMs(account.idToken))) return account.idToken;

	// Stored id token is stale (or undecodable) — mint a new one.
	if (account.refreshToken) {
		const refreshed = await refreshIdToken(account.id, account.refreshToken);
		if (refreshed) return refreshed;
	}

	// Last resort: the access token, but only while it is still valid. An expired
	// one would just fail the exchange downstream with a less obvious error.
	const accessTokenFresh = isFresh(account.accessTokenExpiresAt?.getTime() ?? null);
	return accessTokenFresh && account.accessToken ? account.accessToken : null;
}
