// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { NextResponse } from 'next/server';
import { withPermission } from '@/auth/with-auth';
import {
	CannotResetSelfError,
	createPasswordResetInvitation,
	ResetUserNotFoundError,
} from '@/lib/invitations';

const publicAppUrl = (request: Request): string =>
	process.env.APP_URL?.replace(/\/+$/, '') || new URL(request.url).origin;

// Password reset for an existing user: mint an invite bound to that account.
// The user row stays; redeeming the link updates the credential password.
export const POST = withPermission({ user: ['set-password'] })(async (request, { user }) => {
	let body: { userId?: unknown };
	try {
		body = (await request.json()) as { userId?: unknown };
	} catch {
		return NextResponse.json({ error: 'A JSON body is required.' }, { status: 400 });
	}

	const userId = typeof body.userId === 'string' ? body.userId.trim() : '';
	if (!userId) {
		return NextResponse.json({ error: 'A user id is required.' }, { status: 400 });
	}

	try {
		const invitation = await createPasswordResetInvitation({
			userId,
			createdById: user.id,
			appUrl: publicAppUrl(request),
		});
		return NextResponse.json(
			{
				url: invitation.url,
				email: invitation.email,
				expires_at: invitation.expires_at.toISOString(),
			},
			{ status: 201 },
		);
	} catch (error) {
		if (error instanceof CannotResetSelfError) {
			return NextResponse.json({ error: error.message }, { status: 400 });
		}
		if (error instanceof ResetUserNotFoundError) {
			return NextResponse.json({ error: error.message }, { status: 404 });
		}
		throw error;
	}
});
