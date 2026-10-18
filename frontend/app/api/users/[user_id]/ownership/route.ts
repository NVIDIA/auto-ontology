// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';

type Ctx = { params: Promise<{ user_id: string }> };

// After a role change, drop this account from objects it can no longer own.
export const POST = withPermission<Ctx>({ user: ['set-role'] })((req, { user }) =>
	proxyToBackend(req, { userId: user.id }),
);
