// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';
import { getSessionCookie } from 'better-auth/cookies';

// Pages reachable without a session. Everything else requires authentication.
// `/.well-known` carries the OAuth discovery document an MCP client reads
// before it can sign anyone in, so gating it would deadlock login. `/oauth`
// is the consent and callback-handoff UI for MCP sign-in, same class as /login.
const PUBLIC_PATHS = ['/login', '/.well-known', '/oauth'];

const isPublicPath = (pathname: string): boolean =>
	PUBLIC_PATHS.some((path) => pathname === path || pathname.startsWith(`${path}/`));

/**
 * Optimistic, cookie-only auth gate (no DB call). Redirects unauthenticated
 * users to /login and authenticated users away from the auth pages. Role-based
 * gating of admin areas (e.g. /settings/*) happens in those layouts/pages via
 * requireAdmin(), since the role is not present in the session cookie.
 */
const withPathname = (request: NextRequest): NextResponse => {
	const requestHeaders = new Headers(request.headers);
	requestHeaders.set('x-gsf-pathname', request.nextUrl.pathname);
	return NextResponse.next({ request: { headers: requestHeaders } });
};

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

	return withPathname(request);
}

export function proxy(request: NextRequest) {
	const { pathname } = request.nextUrl;

	// API routes pass straight through: every route handler enforces its own
	// access via withPublic / withPermission (see auth/with-auth.ts), which is
	// the single source of truth. Pages are still gated below, because a route
	// handler cannot redirect the way the middleware can.
	if (pathname.startsWith('/api/')) {
		return withPathname(request);
	}

	return guardPage(request);
}

export const config = {
	// Run on all pages (for the auth gate), excluding Next internals and static
	// assets.
	matcher: ['/((?!_next/static|_next/image|favicon.svg|.*\\.svg$).*)'],
};
