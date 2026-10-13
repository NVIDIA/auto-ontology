// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';
import { withTagAuthors } from '@/lib/tagAuthors';

// termsApi.getById — fetch one Term node. Authors are resolved here so Last
// Modified can name the person who saved it.
export const GET = withPermission({ catalog: ['read'] })(async (req) =>
	withTagAuthors(await proxyToBackend(req)),
);

// Updating a Term name invalidates cached SqlAttribute description suggestions in backend.
export const PATCH = withPermission({ catalog: ['edit'] })((req, { user }) =>
	proxyToBackend(req, { userId: user.id }),
);
