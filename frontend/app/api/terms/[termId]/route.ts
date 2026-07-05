// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';
import { resolveZoneIds } from '@/auth/resolve-zones';

// termsApi.getById — fetch one Term node, zone-scoped for viewers.
//
// This returns a single object (`{ data: TermDetail }`), not a list, so it
// can't reuse proxyToBackend's empty-zone-viewer shortcut (which returns the
// list-shaped `{ data: [], count: 0 }`) — a zero-zone viewer gets a 404
// instead, matching what the backend returns for an out-of-zone/missing term.
export const GET = withPermission({ catalog: ['read'] })(async (req, { user }) => {
	const zoneIds = await resolveZoneIds(user.id, user.role);
	if (zoneIds !== null && zoneIds.length === 0) {
		return new Response(JSON.stringify({ detail: 'Term not found' }), {
			status: 404,
			headers: { 'Content-Type': 'application/json' },
		});
	}
	return proxyToBackend(req, { zoneIds: zoneIds ?? undefined });
});
