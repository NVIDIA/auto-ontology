// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';
import { authorsRequested, withTagAuthors } from '@/lib/tagAuthors';

// tagsApi.getById — one tag. What it labels is `targets/route.ts` beside this.
//
// `manage` rather than the `read` that guards the tag *list*: only the settings
// page asks for a single tag, and that whole section is admin-only already.
//
// Authors on request, by the same rule as the list: this is where the detail
// view reads who curated the tag, and only the gateway can put names to those
// ids.
export const GET = withPermission({ tag: ['manage'] })(async (req, { user }) =>
	authorsRequested(req, user) ? withTagAuthors(await proxyToBackend(req)) : proxyToBackend(req),
);

// tagsApi.update — rename a tag (admin only). `manage` rather than the
// `catalog: ['edit']` the attach routes use: renaming rewrites the vocabulary
// itself, and the new name reaches every object already carrying the tag.
//
// The identity goes down for the backend's `modified_by`, and the renamed tag
// comes back with both authors resolved.
export const PATCH = withPermission({ tag: ['manage'] })(async (req, { user }) =>
	withTagAuthors(await proxyToBackend(req, { userId: user.id })),
);

// tagsApi.delete — remove a tag (admin only). Nothing survives the delete to
// record an author on, so no identity is forwarded.
export const DELETE = withPermission({ tag: ['manage'] })((req) => proxyToBackend(req));
