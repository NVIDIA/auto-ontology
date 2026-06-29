// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { NextResponse } from 'next/server';
import { getCurrentSession } from '@/auth/auth-guards';
import { Role } from '@/enums/auth';

const DISCOVERY_TIMEOUT_MS = 8000;

/**
 * Admin-only helper that fetches an OIDC provider's discovery document
 * server-side and returns the endpoints the SSO form needs.
 *
 * Resolving the endpoints here (instead of letting Better Auth's SSO plugin
 * auto-discover from the issuer) lets us register providers with
 * `skipDiscovery: true`. For public IdPs that path requires no `trustedOrigins`
 * entry, so admins can add any provider without per-provider config.
 */
export async function GET(request: Request) {
	// Admin-only: this is an outbound fetch proxy, so don't expose it to
	// non-admins (the edge middleware only checks for a session cookie).
	const session = await getCurrentSession();
	if (!session || session.user.role !== Role.Admin) {
		return NextResponse.json({ error: 'Forbidden' }, { status: 403 });
	}

	const issuer = new URL(request.url).searchParams.get('issuer')?.trim();
	if (!issuer) {
		return NextResponse.json({ error: 'Missing "issuer" query parameter.' }, { status: 400 });
	}

	let issuerUrl: URL;
	try {
		issuerUrl = new URL(issuer);
	} catch {
		return NextResponse.json({ error: 'Invalid issuer URL.' }, { status: 400 });
	}
	if (issuerUrl.protocol !== 'https:') {
		return NextResponse.json({ error: 'Issuer must use https.' }, { status: 400 });
	}

	const discoveryEndpoint = `${issuer.replace(/\/$/, '')}/.well-known/openid-configuration`;

	const controller = new AbortController();
	const timeout = setTimeout(() => controller.abort(), DISCOVERY_TIMEOUT_MS);
	let doc: Record<string, unknown>;
	try {
		const res = await fetch(discoveryEndpoint, {
			signal: controller.signal,
			headers: { accept: 'application/json' },
		});
		if (!res.ok) {
			return NextResponse.json(
				{ error: `Discovery request failed (HTTP ${res.status}).` },
				{ status: 502 },
			);
		}
		doc = (await res.json()) as Record<string, unknown>;
	} catch (error) {
		const aborted = error instanceof Error && error.name === 'AbortError';
		return NextResponse.json(
			{
				error: aborted
					? 'Discovery request timed out.'
					: 'Could not reach the discovery endpoint.',
			},
			{ status: 504 },
		);
	} finally {
		clearTimeout(timeout);
	}

	// Minimum endpoints required to register and complete an OIDC login.
	for (const key of ['issuer', 'authorization_endpoint', 'token_endpoint'] as const) {
		if (typeof doc[key] !== 'string') {
			return NextResponse.json(
				{ error: `Discovery document is missing "${key}".` },
				{ status: 502 },
			);
		}
	}

	return NextResponse.json({
		issuer: doc.issuer,
		authorizationEndpoint: doc.authorization_endpoint,
		tokenEndpoint: doc.token_endpoint,
		userInfoEndpoint: typeof doc.userinfo_endpoint === 'string' ? doc.userinfo_endpoint : null,
		jwksEndpoint: typeof doc.jwks_uri === 'string' ? doc.jwks_uri : null,
		discoveryEndpoint,
	});
}
