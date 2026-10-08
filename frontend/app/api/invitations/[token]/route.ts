// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { NextResponse } from 'next/server';
import { withPermission } from '@/auth/with-auth';
import { deleteInvitation, InviteNotFoundError } from '@/lib/invitations';

type Ctx = { params: Promise<{ token: string }> };

// Permanently remove an unused invitation by id. An active link stops
// redeeming. Accept already deletes the invite; the user account stays.
export const DELETE = withPermission<Ctx>({ user: ['delete'] })(async (_req, { params }) => {
	const { token: id } = await params;
	if (!id) {
		return NextResponse.json({ error: 'Invitation not found.' }, { status: 404 });
	}

	try {
		await deleteInvitation(id);
		return NextResponse.json({ id });
	} catch (error) {
		if (error instanceof InviteNotFoundError) {
			return NextResponse.json({ error: 'Invitation not found.' }, { status: 404 });
		}
		throw error;
	}
});
