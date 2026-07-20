// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';

// Whether the semantic layer has been calculated (any Term exists). The chat
// page uses this to gate the conversation area when the layer is missing.
export const GET = withPermission({ chat: ['use'] })(async (req) => proxyToBackend(req));
