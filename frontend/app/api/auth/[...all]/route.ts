// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { NextResponse } from 'next/server';
import { toNextJsHandler } from 'better-auth/next-js';
import { auth } from '@/auth/auth';
import { isMcpClientCallback, mcpHandoffLocation } from '@/auth/oauth-loopback';
import { withPublic } from '@/auth/with-auth';

const handlers = toNextJsHandler(auth);

// The apiKey plugin mounts its own management endpoints under this catch-all
// (`/api/auth/api-key/create|get|list|update|delete`). They are session-gated by
// the plugin, but they answer *beside* `app/api/api-tokens`, so they skip both
// the `apiToken: ['manage']` permission and this app's rule that token
// management needs a real session — which would make that permission decorative
// and leave two divergent ways to mint a token.
//
// Blocking them here keeps `/api/api-tokens` the single entry point. Nothing
// internal depends on the HTTP routes: the management handlers call
// `auth.api.createApiKey` / `listApiKeys` / `deleteApiKey` in-process, and
// `verifyApiKey` is server-only and never mounted.
const API_KEY_PREFIX = '/api/auth/api-key';

const blockPluginApiKeyRoutes =
	(handler: (req: Request) => Promise<Response>) =>
	async (req: Request): Promise<Response> => {
		if (new URL(req.url).pathname.startsWith(API_KEY_PREFIX)) {
			return NextResponse.json({ error: 'Not found' }, { status: 404 });
		}
		return handler(req);
	};

// Better Auth's authorize (and HTML consent) responses 302 to the MCP client's
// redirect_uri. For Cursor that is `http://localhost:8787/callback`, a process
// that is gone by the time Chrome paints the tab — "This site can't be reached".
// Send the browser to a Auto Ontology page that delivers the code with fetch instead.
const rewriteMcpClientRedirect =
	(handler: (req: Request) => Promise<Response>) =>
	async (req: Request): Promise<Response> => {
		const response = await handler(req);
		if (response.status < 300 || response.status >= 400) return response;
		const location = response.headers.get('location');
		if (!location) return response;
		let callback: URL;
		try {
			callback = new URL(location, req.url);
		} catch {
			return response;
		}
		if (!isMcpClientCallback(callback, req.url)) return response;
		const headers = new Headers(response.headers);
		headers.set('location', mcpHandoffLocation(callback, req.url));
		return new NextResponse(response.body, {
			status: response.status,
			statusText: response.statusText,
			headers,
		});
	};

// Better Auth's GET surface: session lookup and the SSO callbacks.
//
// Public by design — sign-in and the callback that completes it must both be
// reachable without a session. The plugin's own `/api/auth/api-key/*` routes
// are blocked here so `/api/api-tokens` stays the single way to manage tokens.
export const GET = withPublic(blockPluginApiKeyRoutes(rewriteMcpClientRedirect(handlers.GET)));
// Better Auth's POST surface: sign-in, sign-out, and the SSO callbacks.
//
// Public for the same reason GET is — none of it can require a session. The
// plugin's own `/api/auth/api-key/*` management routes are blocked here so
// `/api/api-tokens` stays the single way to mint or revoke a token.
export const POST = withPublic(blockPluginApiKeyRoutes(rewriteMcpClientRedirect(handlers.POST)));
