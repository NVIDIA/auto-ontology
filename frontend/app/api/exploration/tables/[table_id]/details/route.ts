// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { proxyToBackend } from '@/auth/proxy-backend';
import { withPermission } from '@/auth/with-auth';

// explorationApi.getTableExplorationDetails — columns-adjacent SQL and Term details for one table.
export const GET = withPermission({ catalog: ['read'] })((req) => proxyToBackend(req));
