// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';

// datasources.getSchemasForDatabase — list schemas for one database.
export const GET = withPermission({ catalog: ['read'] })((req) => proxyToBackend(req));
