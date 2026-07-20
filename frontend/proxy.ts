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
 * gating of admin areas (e.g. /settings/*) happens in those layouts/pages via
 * requireAdmin(), since the role is not present in the session cookie.
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

	// NOTE: we deliberately do NOT bounce a cookie-bearing request off /login
	// here. getSessionCookie() only checks presence, not validity — a stale or
	// rotated-secret cookie looks "logged in" to the middleware but is rejected
	// by the authoritative requireUser() on protected pages. Bouncing it to
	// /chat would ping-pong forever (/login → /chat → requireUser → /login).
	// Redirecting an already-authenticated user off /login is instead handled
	// authoritatively on the login page via useSession() (which validates and
	// clears an invalid cookie).

	return NextResponse.next();
}

export function proxy(request: NextRequest) {
	const { pathname } = request.nextUrl;

	// API routes are not auth-gated here: every route handler enforces its own
	// access via withPublic / withPermission (see auth/with-auth.ts), which is
	// the single source of truth. The middleware only performs the `*_id` → path
	// rewrite for /api/*. Pages are still gated below (handlers can't redirect).
	if (pathname.startsWith('/api/')) {
		return rewriteApiIdParam(request);
	}

	return guardPage(request);
}

export const config = {
	// Run on API routes (for the _id rewrite) and all pages (for the auth gate),
	// excluding Next internals and static assets.
	matcher: ['/((?!_next/static|_next/image|favicon.svg|.*\\.svg$).*)'],
};
