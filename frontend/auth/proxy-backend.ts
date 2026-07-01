// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Forward a request to the Python (FastAPI) backend, preserving method, path,
// query, and body. Used by route handlers that need permission gating for
// endpoints that would otherwise be plain next.config rewrites — the wrapper
// (withPermission) enforces access, this just relays the call. The frontend and
// backend share the same `/api/...` path, so we forward the incoming pathname
// verbatim.

const PYTHON_API_URL = process.env.PYTHON_API_URL ?? 'http://127.0.0.1:3001';

export async function proxyToBackend(req: Request): Promise<Response> {
	const incoming = new URL(req.url);
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
