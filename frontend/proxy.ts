// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';
import { getSessionCookie } from 'better-auth/cookies';

// Pages reachable without a session. Everything else requires authentication.
const PUBLIC_PATHS = ['/login'];

const isPublicPath = (pathname: string): boolean =>
	PUBLIC_PATHS.some((path) => pathname === path || pathname.startsWith(`${path}/`));

// API paths reachable without a session: Better Auth's own endpoints (sign-in,
// SSO callback, session), the health check, and the public SSO-provider list
// the login page reads before authenticating. Everything else under /api/*
// requires a session.
const PUBLIC_API_PREFIXES = ['/api/auth', '/api/health', '/api/sso-providers'];

const isPublicApi = (pathname: string): boolean =>
	PUBLIC_API_PREFIXES.some((path) => pathname === path || pathname.startsWith(`${path}/`));

/**
 * For /api/* requests, moves the first `*_id` query parameter into the URL path
 * so the backend receives it as a path param.
 *
 * Example: GET /api/schemas?db_id=abc  →  GET /api/schemas/abc
 */
function rewriteApiIdParam(request: NextRequest): NextResponse {
	const url = request.nextUrl.clone();

	for (const [key, value] of url.searchParams.entries()) {
		if (key.endsWith('_id')) {
			url.pathname = `${url.pathname}/${encodeURIComponent(value)}`;
			url.searchParams.delete(key);
			return NextResponse.rewrite(url);
		}
	}

	return NextResponse.next();
}

/**
 * Optimistic, cookie-only auth gate (no DB call). Redirects unauthenticated
 * users to /login and authenticated users away from the auth pages. Role-based
 * gating of /admin/* happens in the pages themselves via requireAdmin(), since
 * the role is not present in the session cookie.
 */
function guardPage(request: NextRequest): NextResponse {
	const { pathname, search } = request.nextUrl;
	const hasSession = getSessionCookie(request) != null;
	const isPublic = isPublicPath(pathname);

	if (!hasSession && !isPublic) {
		const loginUrl = new URL('/login', request.url);
		loginUrl.searchParams.set('next', `${pathname}${search}`);
		return NextResponse.redirect(loginUrl);
	}

	if (hasSession && isPublic) {
		return NextResponse.redirect(new URL('/chat', request.url));
	}

	return NextResponse.next();
}

export function proxy(request: NextRequest) {
	const { pathname } = request.nextUrl;

	if (pathname.startsWith('/api/')) {
		// Gate API routes: handlers aren't covered by the page gate. This is an
		// optimistic cookie-presence check (the route handlers do the real
		// session validation via requireApiAuth) — reject outright when no
		// session cookie is present, except for the public API allowlist.
		if (!isPublicApi(pathname) && getSessionCookie(request) == null) {
			return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
		}
		return rewriteApiIdParam(request);
	}

	return guardPage(request);
}

export const config = {
	// Run on API routes (for the _id rewrite) and all pages (for the auth gate),
	// excluding Next internals and static assets.
	matcher: ['/((?!_next/static|_next/image|favicon.svg|.*\\.svg$).*)'],
};
