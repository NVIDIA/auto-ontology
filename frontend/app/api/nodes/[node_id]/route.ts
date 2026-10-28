// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';

// datasources.updateNode — edit mutable properties of a catalog node.
// The backend authorizes the write against the account role or this node's
// owner group. Requiring read here admits an owner-group viewer. A column
// stores no owner, so that viewer is still rejected there.
export const PATCH = withPermission({ catalog: ['read'] })((req, { user }) =>
	proxyToBackend(req, { userId: user.id }),
);
