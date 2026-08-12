// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';

// Updating a ColumnAttribute changes only name/description; backend refreshes
// the corresponding semantic VDB embedding.
export const PATCH = withPermission({ catalog: ['edit'] })((req) => proxyToBackend(req));
