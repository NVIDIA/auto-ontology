// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';

// termsApi.list — one page of Terms and their per-card count breakdowns.
// `skip`/`limit` pass straight through.
export const GET = withPermission({ catalog: ['read'] })((req) => proxyToBackend(req));
