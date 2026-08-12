// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';

// Creating a SqlAttribute is a catalog-editing action, available to viewers
// too (catalog:edit) — same permission PATCH /api/nodes/{node_id} uses.
export const POST = withPermission({ catalog: ['edit'] })((req) => proxyToBackend(req));
