// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';

// documentsApi.getAll — list uploaded documents with their ingest status.
export const GET = withPermission({ document: ['read'] })((req) => proxyToBackend(req));

// documentsApi.upload — store a file and ingest it in the background (admin
// only). Multipart; the proxy relays the body as bytes.
export const POST = withPermission({ document: ['manage'] })((req) => proxyToBackend(req));
