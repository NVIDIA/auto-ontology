// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';
import { withTagAuthors } from '@/lib/tagAuthors';

// sqlAttributesApi.get — fetch one SqlAttribute node, with the account behind
// created_by / modified_by so the side panel can name them.
export const GET = withPermission({ catalog: ['read'] })(async (req) =>
	withTagAuthors(await proxyToBackend(req)),
);

// Replace a SqlAttribute's SQL. The backend authorizes the write against the
// account role or this attribute's owner group. Requiring read here admits an
// owner-group viewer without granting writes to any other attribute.
export const PUT = withPermission({ catalog: ['read'] })((req, { user }) =>
	proxyToBackend(req, { userId: user.id }),
);

// Patch name, description, certification, or owner. The backend authorizes
// the write against the account role or this attribute's owner group.
// Requiring read here admits an owner-group viewer.
export const PATCH = withPermission({ catalog: ['read'] })((req, { user }) =>
	proxyToBackend(req, { userId: user.id }),
);

// Delete a SqlAttribute. The backend authorizes the delete against the account
// role or this attribute's owner group. Requiring read here admits an
// owner-group viewer.
export const DELETE = withPermission({ catalog: ['read'] })((req, { user }) =>
	proxyToBackend(req, { userId: user.id }),
);
