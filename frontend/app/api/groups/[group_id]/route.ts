// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';

type Ctx = { params: Promise<{ group_id: string }> };

// groupsApi — one group. The path is forwarded as-is, so the id stays in the URL.
export const GET = withPermission<Ctx>({ group: ['read'] })((req, { user }) =>
	proxyToBackend(req, { userId: user.id }),
);

// groupsApi.update — edit a group or replace its members.
export const PATCH = withPermission<Ctx>({ group: ['manage'] })((req, { user }) =>
	proxyToBackend(req, { userId: user.id }),
);

// groupsApi.delete — remove a group and its membership.
export const DELETE = withPermission<Ctx>({ group: ['manage'] })((req, { user }) =>
	proxyToBackend(req, { userId: user.id }),
);
