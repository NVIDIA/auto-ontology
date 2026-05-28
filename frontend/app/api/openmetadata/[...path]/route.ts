// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { NextRequest, NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

const OM_HOST = process.env.OPENMETADATA_HOST ?? 'http://localhost:8585';
const OM_TOKEN = process.env.OPENMETADATA_TOKEN ?? '';

type Ctx = { params: Promise<{ path: string[] }> };

async function forward(req: NextRequest, ctx: Ctx, method: string): Promise<Response> {
	const { path } = await ctx.params;
	const target = new URL(`${OM_HOST}/api/v1/${path.join('/')}`);
	req.nextUrl.searchParams.forEach((v, k) => target.searchParams.append(k, v));

	const headers: Record<string, string> = {
		Accept: 'application/json',
	};
	if (OM_TOKEN) headers.Authorization = `Bearer ${OM_TOKEN}`;

	const contentType = req.headers.get('content-type');
	if (contentType) headers['Content-Type'] = contentType;

	let body: BodyInit | undefined;
	if (method !== 'GET' && method !== 'HEAD' && method !== 'DELETE') {
		body = await req.arrayBuffer();
	}

	try {
		const upstream = await fetch(target.toString(), {
			method,
			headers,
			body,
			cache: 'no-store',
		});
		const text = await upstream.text();
		const respHeaders = new Headers();
		const ct = upstream.headers.get('content-type');
		if (ct) respHeaders.set('content-type', ct);
		return new NextResponse(text, { status: upstream.status, headers: respHeaders });
	} catch (err) {
		const detail = err instanceof Error ? err.message : String(err);
		return NextResponse.json(
			{
				error: 'openmetadata_unreachable',
				detail,
				target: target.toString(),
				hint: 'Is OpenMetadata running at OPENMETADATA_HOST? Did you set OPENMETADATA_TOKEN?',
			},
			{ status: 502 },
		);
	}
}

export const GET = (req: NextRequest, ctx: Ctx) => forward(req, ctx, 'GET');
export const POST = (req: NextRequest, ctx: Ctx) => forward(req, ctx, 'POST');
export const PUT = (req: NextRequest, ctx: Ctx) => forward(req, ctx, 'PUT');
export const PATCH = (req: NextRequest, ctx: Ctx) => forward(req, ctx, 'PATCH');
export const DELETE = (req: NextRequest, ctx: Ctx) => forward(req, ctx, 'DELETE');
