// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';

// tagsApi.detach — take a tag off one object. Guarded as the attach route is:
// removing a label edits the object, not the tag.
export const DELETE = withPermission({ catalog: ['edit'] })((req) => proxyToBackend(req));
