// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';

// datasources.updateNode — edit mutable properties of a catalog node
// (table/column descriptions, etc.). Allowed for viewers and admins.
export const PATCH = withPermission({ catalog: ['edit'] })((req) => proxyToBackend(req));
