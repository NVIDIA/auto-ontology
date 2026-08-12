// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';

// modelInterchangeApi.importModel — upload a native GSF or Apache Ossie model
// YAML file and apply it to the catalog + semantic layer. Admin-only: can
// replace existing data.
export const POST = withPermission({ modelInterchange: ['import'] })((req) => proxyToBackend(req));
