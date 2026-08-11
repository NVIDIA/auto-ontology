// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';

// connectionsApi.setSsoFederation — toggle "authenticate as signed-in user"
// on an existing connection (admin only).
export const PATCH = withPermission({ connection: ['manage'] })((req) => proxyToBackend(req));
