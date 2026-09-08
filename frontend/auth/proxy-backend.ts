// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Forward a request to the Python (FastAPI) backend, preserving method, path,
// query, and body. Used by route handlers that need permission gating for
// endpoints that would otherwise be plain next.config rewrites — the wrapper
// (withPermission) enforces access, this just relays the call. The frontend and
// backend share the same `/api/...` path, so we forward the incoming pathname
// verbatim.
//
// Zone scoping is not applied here: zone membership stopped being an
// authorization boundary, so the backend's own `zone_ids` default (unscoped)
// is what every proxied call gets.

import { buildInternalIdentityHeaders } from '@/lib/internalIdentity';

const PYTHON_API_URL = process.env.PYTHON_API_URL ?? 'http://127.0.0.1:3001';

/**
 * What every proxied answer is allowed to be kept in, which is nothing shared.
 *
 * Every route that reaches this proxy is behind `withPermission`, so every
 * answer it relays is scoped to the caller who asked for it — `/api/tags` most
 * literally, since it carries author names only for a caller who may manage
 * tags. Next does not cache Route Handlers, so this is about the browser: a GET
 * that states no policy is left to heuristic freshness, and none of these
 * answers may be handed to the next person to use the machine.
 *
 * Exported because a wrapper that rebuilds one of these responses has to carry
 * it through rather than decide it again — see `withTagAuthors`.
 */
export const PROXY_CACHE_CONTROL = 'private, no-store';

/**
 * `userId` names the caller to the backend, for routes that record who did
 * something (a tag's `created_by`, for instance). Opt-in rather than always
 * forwarded, so a route that has no use for an identity does not imply one.
 *
 * It comes from `withPermission`'s already-resolved user, and the header set is
 * rebuilt from scratch below — so a client cannot smuggle in an identity of its
 * own choosing by sending the header itself.
 */
export async function proxyToBackend(
	req: Request,
	{ userId }: { userId?: string } = {},
): Promise<Response> {
	const incoming = new URL(req.url);
	const target = `${PYTHON_API_URL}${incoming.pathname}${incoming.search}`;

	const headers: Record<string, string> = { Accept: 'application/json' };
	const contentType = req.headers.get('content-type');
	if (contentType) headers['Content-Type'] = contentType;
	if (userId) Object.assign(headers, buildInternalIdentityHeaders(userId));

	const hasBody = req.method !== 'GET' && req.method !== 'HEAD';
	const body = hasBody ? await req.text() : undefined;

	const upstream = await fetch(target, { method: req.method, headers, body });

	const respBody = await upstream.text();
	return new Response(respBody, {
		status: upstream.status,
		headers: {
			'Content-Type': upstream.headers.get('content-type') ?? 'application/json',
			'Cache-Control': PROXY_CACHE_CONTROL,
		},
	});
}
