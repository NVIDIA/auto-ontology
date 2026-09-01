// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';

// LLM-assisted description suggestion is only ever surfaced in the edit flow
// for a SqlAttribute, so it shares the same permission as saving one.
export const GET = withPermission({ catalog: ['edit'] })((req) => proxyToBackend(req));
