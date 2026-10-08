// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { NextResponse } from 'next/server';
import { withPublic } from '@/auth/with-auth';
import { acceptInvitation } from '@/lib/invitations';

type RouteContext = { params: Promise<{ token: string }> };

// Redeem an invite: set a password on the invited email and create the account.
// Public — the unguessable token is the credential. The link is one-shot.
export const POST = withPublic<RouteContext>(async (request, { params }) => {
	const { token } = await params;
	if (!token) {
		return NextResponse.json(
			{ error: 'This invitation is invalid or has expired.' },
			{ status: 404 },
		);
	}

	let body: { password?: unknown };
	try {
		body = (await request.json()) as { password?: unknown };
	} catch {
		return NextResponse.json({ error: 'A JSON body is required.' }, { status: 400 });
	}

	const password = typeof body.password === 'string' ? body.password : '';
	if (!password) {
		return NextResponse.json({ error: 'A password is required.' }, { status: 400 });
	}

	const result = await acceptInvitation(token, password);
	if ('error' in result) {
		return NextResponse.json({ error: result.error }, { status: result.status });
	}
	return NextResponse.json({ email: result.email });
});
