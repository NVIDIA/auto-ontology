// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';

// graphApi.status — whether the optional Apache AGE projection is on and built.
export const GET = withPermission({ catalog: ['read'] })((req) => proxyToBackend(req));
