// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { NextResponse } from 'next/server';
import { withPermission } from '@/auth/with-auth';
import {
	createInvitation,
	InviteConflictError,
	listInvitations,
	parseInviteRole,
} from '@/lib/invitations';

const publicAppUrl = (request: Request): string =>
	process.env.APP_URL?.replace(/\/+$/, '') || new URL(request.url).origin;

const isEmail = (value: string): boolean => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value);

// List unused invites (active or expired). Accepted rows are deleted on
// redeem, so they never appear here. Active rows include `url`.
export const GET = withPermission({ user: ['list'] })(async (request) => {
	const invitations = await listInvitations(publicAppUrl(request));
	return NextResponse.json({ invitations });
});

// Create an email-bound invite link. The token is returned in `url` and kept
// while the invite is active so the list can copy it later. No email is sent.
// Replaces any unused invite for the same address.
export const POST = withPermission({ user: ['create'] })(async (request, { user }) => {
	let body: { email?: unknown; name?: unknown; role?: unknown };
	try {
		body = (await request.json()) as { email?: unknown; name?: unknown; role?: unknown };
	} catch {
		return NextResponse.json({ error: 'A JSON body is required.' }, { status: 400 });
	}

	const email = typeof body.email === 'string' ? body.email.trim().toLowerCase() : '';
	if (!email || !isEmail(email)) {
		return NextResponse.json({ error: 'A valid email is required.' }, { status: 400 });
	}

	const name = typeof body.name === 'string' ? body.name.trim() : '';
	const role = parseInviteRole(body.role);
	if (!role) {
		return NextResponse.json({ error: 'Role must be admin or viewer.' }, { status: 400 });
	}

	try {
		const invitation = await createInvitation({
			email,
			name,
			role,
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
		if (error instanceof InviteConflictError) {
			return NextResponse.json({ error: error.message }, { status: 409 });
		}
		throw error;
	}
});
