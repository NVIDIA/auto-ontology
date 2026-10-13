// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';

// Creating a SqlAttribute is a catalog edit and requires catalog:edit, the same
// permission PATCH /api/nodes/{node_id} uses. A viewer does not have it.
export const POST = withPermission({ catalog: ['edit'] })((req, { user }) =>
	proxyToBackend(req, { userId: user.id }),
);
