// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';

// usersApi.getAll — list User nodes (admin only).
export const GET = withPermission({ zone: ['manage'] })((req) => proxyToBackend(req));
// usersApi.upsert — create or update a User node in the graph. Called on every
// login/session refresh for any authenticated user, so the permission is intentionally
// permissive (catalog:read is held by both admin and viewer roles).
export const POST = withPermission({ catalog: ['read'] })((req) => proxyToBackend(req));
