// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Self-service management of the caller's own API tokens. Every role may manage
// its own tokens (`apiToken: ['manage']`); the Better Auth apiKey plugin scopes
// list/create to the session user, so one user can never see or mint another's.

import { headers } from 'next/headers';
import { NextResponse } from 'next/server';
import { auth } from '@/auth/auth';
import { withPermission } from '@/auth/with-auth';
import {
	apiTokenErrorResponse,
	requireSessionCaller,
	toApiToken,
	MAX_EXPIRES_IN_DAYS,
	MAX_NAME_LENGTH,
} from '@/lib/apiTokens';

const SECONDS_PER_DAY = 60 * 60 * 24;

export const GET = withPermission({ apiToken: ['manage'] })(async () => {
	const denied = await requireSessionCaller();
	if (denied) return denied;

	const { apiKeys } = await auth.api.listApiKeys({ headers: await headers() });
	return NextResponse.json(apiKeys.map(toApiToken));
});

export const POST = withPermission({ apiToken: ['manage'] })(async (req) => {
	const denied = await requireSessionCaller();
	if (denied) return denied;

	const body = (await req.json()) as { name?: unknown; expires_in_days?: unknown };

	const name = typeof body.name === 'string' ? body.name.trim() : '';
	if (!name) return NextResponse.json({ error: 'A token name is required.' }, { status: 400 });
	if (name.length > MAX_NAME_LENGTH) {
		return NextResponse.json(
			{ error: `A token name may be at most ${MAX_NAME_LENGTH} characters.` },
			{ status: 400 },
		);
	}

	// `expires_in_days` is optional — omitting it mints a non-expiring token,
	// which is what an unattended script usually wants.
	const days = body.expires_in_days;
	if (days != null && (typeof days !== 'number' || !Number.isFinite(days))) {
		return NextResponse.json({ error: 'expires_in_days must be a number.' }, { status: 400 });
	}
	if (typeof days === 'number' && (days < 1 || days > MAX_EXPIRES_IN_DAYS)) {
		return NextResponse.json(
			{ error: `expires_in_days must be between 1 and ${MAX_EXPIRES_IN_DAYS}.` },
			{ status: 400 },
		);
	}

	// The checks above cover what this route promises, but the plugin enforces
	// rules of its own; relay those rather than letting them surface as a 500.
	let created;
	try {
		created = await auth.api.createApiKey({
			body: {
				name,
				expiresIn: typeof days === 'number' ? days * SECONDS_PER_DAY : null,
			},
			headers: await headers(),
		});
	} catch (err) {
		const mapped = apiTokenErrorResponse(err);
		if (mapped) return mapped;
		throw err;
	}

	// The only moment the plaintext token exists outside the caller's script —
	// it is stored hashed, so it can never be shown again.
	return NextResponse.json({ ...toApiToken(created), token: created.key }, { status: 201 });
});
