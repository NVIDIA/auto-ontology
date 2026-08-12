// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';

// sqlAttributesApi.get — fetch one SqlAttribute node.
export const GET = withPermission({ catalog: ['read'] })((req) => proxyToBackend(req));

// Updating a SqlAttribute re-parses SQL and refreshes its embedding in backend.
export const PUT = withPermission({ catalog: ['edit'] })((req) => proxyToBackend(req));

// Patching name/description does not touch SQL expression.
export const PATCH = withPermission({ catalog: ['edit'] })((req) => proxyToBackend(req));

// Deleting a SqlAttribute mutates catalog data and the backend handles graph
// cleanup plus VDB embedding deletion.
export const DELETE = withPermission({ catalog: ['edit'] })((req) => proxyToBackend(req));
