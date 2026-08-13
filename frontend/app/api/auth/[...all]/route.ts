// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { NextResponse } from 'next/server';
import { toNextJsHandler } from 'better-auth/next-js';
import { auth } from '@/auth/auth';
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

// Better Auth's GET surface: session lookup and the SSO callbacks.
//
// Public by design — sign-in and the callback that completes it must both be
// reachable without a session. The plugin's own `/api/auth/api-key/*` routes
// are blocked here so `/api/api-tokens` stays the single way to manage tokens.
export const GET = withPublic(blockPluginApiKeyRoutes(handlers.GET));
// Better Auth's POST surface: sign-in, sign-out, and the SSO callbacks.
//
// Public for the same reason GET is — none of it can require a session. The
// plugin's own `/api/auth/api-key/*` management routes are blocked here so
// `/api/api-tokens` stays the single way to mint or revoke a token.
export const POST = withPublic(blockPluginApiKeyRoutes(handlers.POST));
