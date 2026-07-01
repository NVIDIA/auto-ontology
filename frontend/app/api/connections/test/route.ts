// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';

// connectionsApi.test — validate a connection config (admin only).
export const POST = withPermission({ connection: ['manage'] })((req) => proxyToBackend(req));
