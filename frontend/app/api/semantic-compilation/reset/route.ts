// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';

// semanticCompilationApi.reset — delete the compiled semantic layer. Scoped to
// one database when `database_name` is given, every database otherwise.
// Destructive, so it sits behind `manage` rather than the `read` tier that
// may view the toggle.
export const POST = withPermission({ semanticCompilation: ['manage'] })((req) =>
	proxyToBackend(req),
);
