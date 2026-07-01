// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { NextResponse } from 'next/server';
import { getPrisma } from '@/lib/prisma';
import { withPublic } from '@/auth/with-auth';

/**
 * Public list of registered SSO providers (no secrets) so the login page can
 * decide whether to show the "Sign in with SSO" option.
 */
export const GET = withPublic(async () => {
	const providers = await getPrisma().ssoProvider.findMany({
		select: { providerId: true, issuer: true, domain: true },
	});
	return NextResponse.json({ providers });
});
