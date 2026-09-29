// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Shared helpers for the /api/api-tokens routes.

import { NextResponse } from 'next/server';
import { APIError } from 'better-auth/api';
import { getCurrentSession } from '@/auth/auth-guards';

/** Longest expiry a token may be minted with (mirrors `keyExpiration.maxExpiresIn`). */
export const MAX_EXPIRES_IN_DAYS = 365;

/**
 * Longest name a token may carry. Mirrors the plugin's `maximumNameLength`, so
 * an over-long name is rejected here with a 400 that names the limit rather than
 * surfacing as the plugin's opaque error.
 */
export const MAX_NAME_LENGTH = 64;

/**
 * Token metadata safe to return to a client. Deliberately omits `key`, which
 * holds the token's hash, and the plugin's rate-limit/refill bookkeeping, which
 * Auto Ontology does not use.
 *
 * Snake_case because that is what the API publishes; the Better Auth model
 * behind it is camelCase, so `toApiToken` maps at the boundary rather than
 * leaking the library's naming (same as `GET /api/sso-providers`).
 */
export type ApiTokenSummary = {
	id: string;
	name: string | null;
	/** First characters of the token (prefix included), to identify it in a list. */
	start: string | null;
	enabled: boolean;
	created_at: string;
	expires_at: string | null;
	last_request: string | null;
};

type ApiKeyRecord = {
	id: string;
	name?: string | null;
	start?: string | null;
	enabled: boolean;
	createdAt: Date | string;
	expiresAt?: Date | string | null;
	lastRequest?: Date | string | null;
};

const iso = (value: Date | string | null | undefined): string | null => {
	if (value == null) return null;
	return value instanceof Date ? value.toISOString() : new Date(value).toISOString();
};

/** Project a plugin API-key record down to the fields a client may see. */
export const toApiToken = (key: ApiKeyRecord): ApiTokenSummary => ({
	id: key.id,
	name: key.name ?? null,
	start: key.start ?? null,
	enabled: key.enabled,
	created_at: iso(key.createdAt) ?? new Date(0).toISOString(),
	expires_at: iso(key.expiresAt),
	last_request: iso(key.lastRequest),
});

/**
 * Turn an error thrown by the apiKey plugin into the response it describes.
 *
 * The plugin signals client mistakes with an `APIError` carrying its own status
 * — 400 for a name past `maximumNameLength`, 404 for a key that is missing or
 * owned by someone else, 401 for a banned user — so relaying that status keeps
 * the cause visible instead of collapsing everything into one code. Returns
 * `null` for anything else, including the plugin's own 5xx (a failed delete, for
 * instance): those are genuine faults and must keep propagating, so they get
 * logged rather than being reported to the caller as their mistake. Telling
 * someone "token not found" when the delete actually failed would leave a live
 * token behind that they believe is revoked.
 */
export function apiTokenErrorResponse(err: unknown): NextResponse | null {
	if (!(err instanceof APIError)) return null;
	const status = typeof err.statusCode === 'number' ? err.statusCode : 500;
	if (status >= 500) return null;
	return NextResponse.json({ error: err.body?.message ?? err.message }, { status });
}

/**
 * Require that the caller is a signed-in browser session rather than a script
 * authenticating with an API token.
 *
 * `withPermission` accepts any credential, but a token must not be able to mint
 * or revoke tokens: that would let a leaked token quietly issue itself
 * successors and outlive the revocation of the original. Returns a 403 response
 * to bail out with, or null when the caller has a real session.
 */
export async function requireSessionCaller(): Promise<NextResponse | null> {
	const session = await getCurrentSession();
	if (session?.user.id) return null;
	return NextResponse.json(
		{ error: 'API tokens must be managed from a signed-in session, not with an API token.' },
		{ status: 403 },
	);
}
