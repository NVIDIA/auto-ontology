// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';

// zonesApi.getAll — list zones (viewable by all).
export const GET = withPermission({ zone: ['read'] })((req) => proxyToBackend(req));
// zonesApi.create — add a zone (admin only).
export const POST = withPermission({ zone: ['manage'] })((req) => proxyToBackend(req));
