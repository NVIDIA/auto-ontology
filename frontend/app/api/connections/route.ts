// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';

// connectionsApi.getAll — list connections (viewable by all).
export const GET = withPermission({ connection: ['read'] })((req) => proxyToBackend(req));
// connectionsApi.create — add a connection (admin only; carries credentials).
export const POST = withPermission({ connection: ['manage'] })((req) => proxyToBackend(req));
