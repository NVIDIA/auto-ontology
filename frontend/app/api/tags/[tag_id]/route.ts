// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';
import { authorsRequested, withTagAuthors } from '@/lib/tagAuthors';

// tagsApi.getById — one tag with the objects it labels.
//
// `manage` rather than the `read` that guards the tag *list*: this answer names
// catalog objects and spells out where they sit, and it is not scoped to the
// caller's zones the way the catalog reads are. Only the settings page asks for
// it, and that whole section is admin-only already.
// Authors on request, by the same rule as the list: the detail view renders a
// tag's name and the objects carrying it, so nothing here reads them today, and
// a route that resolved them anyway would be work with no reader.
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
