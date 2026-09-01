// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { headers } from 'next/headers';
import { NextResponse } from 'next/server';
import { auth } from '@/auth/auth';
import { withPermission } from '@/auth/with-auth';
import { apiTokenErrorResponse, requireSessionCaller } from '@/lib/apiTokens';

type Ctx = { params: Promise<{ id: string }> };

/**
 * Revoke a token. Deletion is immediate and irreversible — the next request
 * carrying that token fails to resolve a user and gets a 401. The plugin scopes
 * the delete to the session user, so `id` can only ever name the caller's own
 * token; an id belonging to someone else reads back as "not found".
 */
export const DELETE = withPermission<Ctx>({ apiToken: ['manage'] })(async (_req, { params }) => {
	const denied = await requireSessionCaller();
	if (denied) return denied;

	const { id } = await params;

	try {
		await auth.api.deleteApiKey({ body: { keyId: id }, headers: await headers() });
	} catch (err) {
		// A 404 here means "no such token of yours"; a failed delete is a 5xx and
		// must keep propagating, so the caller is never told a token is gone
		// while it is still live.
		const mapped = apiTokenErrorResponse(err);
		if (mapped) return mapped;
		throw err;
	}

	return NextResponse.json({ id });
});
