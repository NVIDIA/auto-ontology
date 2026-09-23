// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// API-token (machine-to-machine) authentication.
//
// A user mints a long-lived token from the UI and puts it in a script; the
// script calls the same `/api/...` endpoints the browser does, with no session
// and no interactive login. The token is stored only as a SHA-256 hash (Better
// Auth's apiKey plugin), so verification is a hash lookup — the plaintext is
// unrecoverable after creation.
//
// A token carries no permissions of its own: it resolves to its owning user,
// and the request is then authorized against that user's role by the same
// userCan() check every browser request goes through. Revoking a token or
// deleting its owner therefore revokes the script's access immediately.

import { auth } from '@/auth/auth';
import { getPrisma } from '@/lib/prisma';

/** Prefix every Auto Ontology-minted token carries (see the apiKey plugin config). */
export const API_TOKEN_PREFIX = 'auto_ontology_';

/** Canonical header for API-token auth. */
const API_TOKEN_HEADER = 'x-api-key';

/**
 * Pull a Auto Ontology API token out of the request headers.
 *
 * `x-api-key` is the canonical spelling, but `Authorization: Bearer` is also
 * accepted because it is what most HTTP clients reach for by default.
 *
 * Both headers are claimed only for values carrying our prefix. `Authorization`
 * is shared with SSO bearer tokens, and `x-api-key` is a header gateways
 * routinely stamp with their own credential — claiming any non-empty value made
 * `resolveUser` treat such a request as offering a Auto Ontology token, so a valid SSO id
 * token alongside it produced a hard 401 instead of falling through to the JWT
 * path.
 */
export function extractApiToken(headers: Headers): string | null {
	const direct = headers.get(API_TOKEN_HEADER)?.trim();
	if (direct) return direct.startsWith(API_TOKEN_PREFIX) ? direct : null;

	const authorization = headers.get('authorization');
	if (!authorization) return null;
	const [scheme, value] = authorization.split(' ');
	if (scheme?.toLowerCase() !== 'bearer') return null;

	const token = value?.trim();
	return token?.startsWith(API_TOKEN_PREFIX) ? token : null;
}

/**
 * Verify an API token from the request headers and resolve its owner.
 *
 * Returns the owning user's id and role, or null when there is no token, the
 * token is unknown/disabled/expired, or its owner no longer exists.
 */
export async function verifyApiToken(
	headers: Headers,
): Promise<{ id: string; role: string | null } | null> {
	const token = extractApiToken(headers);
	if (!token) return null;

	const { valid, key } = await auth.api.verifyApiKey({ body: { key: token } });
	if (!valid || !key) return null;

	// `referenceId` is the owner's user id — the plugin is configured with the
	// default references: "user". Re-read the role rather than trusting anything
	// stored on the token, so a role change (or a ban) takes effect immediately.
	const user = await getPrisma().user.findUnique({
		where: { id: key.referenceId },
		select: { id: true, role: true, banned: true },
	});
	if (!user || user.banned) return null;

	return { id: user.id, role: user.role };
}
