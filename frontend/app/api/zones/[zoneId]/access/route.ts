// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';

// zonesApi.listAccess — list users with access to a zone (admin only).
export const GET = withPermission({ zone: ['manage'] })((req) => proxyToBackend(req));
// zonesApi.grantAccess — grant a user access to a zone (admin only).
export const POST = withPermission({ zone: ['manage'] })((req) => proxyToBackend(req));
