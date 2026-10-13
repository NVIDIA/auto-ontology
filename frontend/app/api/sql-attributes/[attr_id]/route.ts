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

// Updating a SqlAttribute re-parses SQL and refreshes its embedding in backend.
export const PUT = withPermission({ catalog: ['edit'] })((req, { user }) =>
	proxyToBackend(req, { userId: user.id }),
);

// Patching name/description does not touch SQL expression.
export const PATCH = withPermission({ catalog: ['edit'] })((req, { user }) =>
	proxyToBackend(req, { userId: user.id }),
);

// Deleting a SqlAttribute mutates catalog data and the backend handles graph
// cleanup plus VDB embedding deletion.
export const DELETE = withPermission({ catalog: ['edit'] })((req, { user }) =>
	proxyToBackend(req, { userId: user.id }),
);
