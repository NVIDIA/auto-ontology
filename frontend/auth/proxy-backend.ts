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
// Pass `options.zoneIds` to inject `?zone_ids=<id>` query params so the Python
// backend can apply zone-based data filtering for viewer accounts.

const PYTHON_API_URL = process.env.PYTHON_API_URL ?? 'http://127.0.0.1:3001';

export type ProxyOptions = {
	/** When set, appended as repeated `?zone_ids=<id>` params for zone-scoped filtering. */
	zoneIds?: string[];
	/** Response body returned when a viewer has no zones. */
	emptyResponse?: unknown;
};

export async function proxyToBackend(req: Request, options?: ProxyOptions): Promise<Response> {
	// Viewer with no zone access: short-circuit and return an empty result
	// rather than forwarding the request without zone_ids (which would make
	// the backend return all data, effectively treating the viewer as admin).
	if (options?.zoneIds !== undefined && options.zoneIds.length === 0) {
		return new Response(JSON.stringify(options.emptyResponse ?? { data: [], count: 0 }), {
			headers: { 'Content-Type': 'application/json' },
		});
	}

	const incoming = new URL(req.url);
	if (options?.zoneIds?.length) {
		for (const id of options.zoneIds) {
			incoming.searchParams.append('zone_ids', id);
		}
	}
	const target = `${PYTHON_API_URL}${incoming.pathname}${incoming.search}`;

	const headers: Record<string, string> = { Accept: 'application/json' };
	const contentType = req.headers.get('content-type');
	if (contentType) headers['Content-Type'] = contentType;

	const hasBody = req.method !== 'GET' && req.method !== 'HEAD';
	const body = hasBody ? await req.text() : undefined;

	const upstream = await fetch(target, { method: req.method, headers, body });

	const respBody = await upstream.text();
	return new Response(respBody, {
		status: upstream.status,
		headers: { 'Content-Type': upstream.headers.get('content-type') ?? 'application/json' },
	});
}
