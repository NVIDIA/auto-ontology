// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';

// zonesApi.setEnabled — enable/disable a zone (admin only).
export const PATCH = withPermission({ zone: ['manage'] })((req) => proxyToBackend(req));
