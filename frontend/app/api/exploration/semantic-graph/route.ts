// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';

// explorationApi.getSemanticExplorationGraph — full semantic-layer graph ({nodes, links}).
export const GET = withPermission({ catalog: ['read'] })((req) => proxyToBackend(req));
