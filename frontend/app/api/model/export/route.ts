// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';

// modelInterchangeApi.exportModel — download the catalog + semantic layer as a
// YAML file. Admin-only: it can include every connection's schema.
export const POST = withPermission({ modelInterchange: ['export'] })((req) => proxyToBackend(req));
