// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';
import { resolveZoneIds } from '@/auth/resolve-zones';

// Anyone may list custom analyses; only admins (analysis:manage) may add one.
export const GET = withPermission({ analysis: ['read'] })(async (req, { user }) => {
	const zoneIds = await resolveZoneIds(user.id, user.role);
	return proxyToBackend(req, { zoneIds: zoneIds ?? undefined });
});
export const POST = withPermission({ analysis: ['manage'] })((req) => proxyToBackend(req));
