// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';

// LLM-assisted description suggestion, offered from the SqlAttribute edit
// flow. Catalog read admits an owner-group viewer; the backend allows the
// call only when that viewer may edit this attribute. The generated text is
// cached on the attribute, so the caller's identity goes with the request.
export const GET = withPermission({ catalog: ['read'] })((req, { user }) =>
	proxyToBackend(req, { userId: user.id }),
);
