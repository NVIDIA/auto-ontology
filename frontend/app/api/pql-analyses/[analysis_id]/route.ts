// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';

// Editing/deleting a PQL analysis is admin-only (analysis:manage).
export const PUT = withPermission({ analysis: ['manage'] })((req) => proxyToBackend(req));
export const DELETE = withPermission({ analysis: ['manage'] })((req) => proxyToBackend(req));
