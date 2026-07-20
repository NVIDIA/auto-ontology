// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';
import { resolveZoneIds } from '@/auth/resolve-zones';

// termsApi.getAllWithAttributes — list every term available to the user, each
// with its merged attributes (ColumnAttributes + SqlAttributes), zone-scoped
// for viewers.
export const GET = withPermission({ catalog: ['read'] })(async (req, { user }) => {
	const zoneIds = await resolveZoneIds(user.id, user.role);
	// The backend returns a bare list, so a zero-zone viewer gets `[]` (not the
	// default `{ data: [], count: 0 }`) to keep the response shape consistent.
	return proxyToBackend(req, { zoneIds: zoneIds ?? undefined, emptyResponse: [] });
});
