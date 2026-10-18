// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';

// groupsApi.getAll — list groups. Admin only: each group names its members.
export const GET = withPermission({ group: ['read'] })((req, { user }) =>
	proxyToBackend(req, { userId: user.id }),
);

// groupsApi.create — add a group. The identity header is what the backend
// checks for the admin role, so it has to be the caller this gate just resolved.
export const POST = withPermission({ group: ['manage'] })((req, { user }) =>
	proxyToBackend(req, { userId: user.id }),
);
