// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';

// Anyone may list custom analyses; only admins (analysis:manage) may add one.
export const GET = withPermission({ analysis: ['read'] })((req) => proxyToBackend(req));
export const POST = withPermission({ analysis: ['manage'] })((req) => proxyToBackend(req));
