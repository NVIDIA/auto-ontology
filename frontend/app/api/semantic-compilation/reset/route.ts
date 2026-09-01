// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';

// semanticCompilationApi.reset — delete every database's compiled semantic
// layer. Destructive, so it sits behind `manage` rather than the `read` tier
// that may view the toggle.
export const POST = withPermission({ semanticCompilation: ['manage'] })((req) =>
	proxyToBackend(req),
);
