// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';

// Grade how well the semantic layer covers the entities in a free-text
// question. Same retrieval capability as chat, so it is gated on
// `chat: ['use']`.
export const POST = withPermission({ chat: ['use'] })((req) => proxyToBackend(req));
