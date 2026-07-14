// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';
import { resolveZoneIds } from '@/auth/resolve-zones';

// sqlAttributesApi.get — fetch one SqlAttribute node, zone-scoped for viewers.
//
// This returns a single object (`{ data: SqlAttribute }`), not a list, so a
// zero-zone viewer gets a 404 rather than an empty list — matching the backend
// behaviour for an out-of-zone/missing attribute.
export const GET = withPermission({ catalog: ['read'] })(async (req, { user }) => {
	const zoneIds = await resolveZoneIds(user.id, user.role);
	if (zoneIds !== null && zoneIds.length === 0) {
		return new Response(JSON.stringify({ detail: 'SqlAttribute not found' }), {
			status: 404,
			headers: { 'Content-Type': 'application/json' },
		});
	}
	return proxyToBackend(req, { zoneIds: zoneIds ?? undefined });
});

// Updating a SqlAttribute re-parses SQL and refreshes its embedding in backend.
export const PUT = withPermission({ catalog: ['edit'] })((req) => proxyToBackend(req));

// Patching name/description does not touch SQL expression.
export const PATCH = withPermission({ catalog: ['edit'] })((req) => proxyToBackend(req));

// Deleting a SqlAttribute mutates catalog data and the backend handles graph
// cleanup plus VDB embedding deletion.
export const DELETE = withPermission({ catalog: ['edit'] })((req) => proxyToBackend(req));
