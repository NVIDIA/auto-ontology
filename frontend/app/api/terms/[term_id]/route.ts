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

// The backend authorizes the write against either the account role or this
// term's owner group. Requiring read here admits authenticated owner-group
// viewers without granting them writes to terms owned by anyone else.
export const PATCH = withPermission({ catalog: ['read'] })((req, { user }) =>
	proxyToBackend(req, { userId: user.id }),
);
