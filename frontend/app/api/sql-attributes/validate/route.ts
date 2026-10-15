// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';

// Validate a SQL expression without writing it. Catalog read is enough:
// creating an attribute still requires catalog:edit, and saving one is
// authorized on that attribute.
export const POST = withPermission({ catalog: ['read'] })((req) => proxyToBackend(req));
