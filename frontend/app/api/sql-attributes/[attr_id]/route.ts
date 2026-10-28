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

// Replace a SqlAttribute's SQL. The page shows the term's owner. The account
// role authorizes the write, the same way a column attribute edit does.
export const PUT = withPermission({ catalog: ['edit'] })((req, { user }) =>
	proxyToBackend(req, { userId: user.id }),
);

// Patch name, description, or certification. The account role authorizes the
// write. The term owner shown beside the attribute does not grant it.
export const PATCH = withPermission({ catalog: ['edit'] })((req, { user }) =>
	proxyToBackend(req, { userId: user.id }),
);

// Delete a SqlAttribute. The account role authorizes the delete.
export const DELETE = withPermission({ catalog: ['edit'] })((req, { user }) =>
	proxyToBackend(req, { userId: user.id }),
);
