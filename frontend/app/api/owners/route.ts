// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';

// ownersApi.list — users and groups a picker can offer as an owner.
// Readable by anyone who can open the catalog: a viewer has to see who owns
// a table. The identity goes down so a group's member list is only the
// caller, when they belong to it.
export const GET = withPermission({ catalog: ['read'] })((req, { user }) =>
	proxyToBackend(req, { userId: user.id }),
);
