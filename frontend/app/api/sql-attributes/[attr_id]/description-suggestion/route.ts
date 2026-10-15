// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';

// LLM-assisted description suggestion, offered from the SqlAttribute edit
// flow. The account role must allow a catalog edit, the same way editing the
// attribute does. The generated text is cached on the attribute, so the
// caller's identity goes with the request.
export const GET = withPermission({ catalog: ['edit'] })((req, { user }) =>
	proxyToBackend(req, { userId: user.id }),
);
