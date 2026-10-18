// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';

// Validate a SQL expression without writing it. Catalog read is enough.
// Creating an attribute and saving one both require catalog:edit.
export const POST = withPermission({ catalog: ['read'] })((req) => proxyToBackend(req));
