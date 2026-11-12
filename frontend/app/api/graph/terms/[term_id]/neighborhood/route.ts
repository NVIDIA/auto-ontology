// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';

// graphApi.termNeighborhood — vertices and edges around one Term in the AGE
// projection. Read-only; the traversal is fixed server-side.
export const GET = withPermission({ catalog: ['read'] })((req) => proxyToBackend(req));
