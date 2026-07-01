// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';

// usersApi.getById — fetch one User node by id (admin only).
export const GET = withPermission({ zone: ['manage'] })((req) => proxyToBackend(req));
