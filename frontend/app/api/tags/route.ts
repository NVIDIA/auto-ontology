// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';
import { authorsRequested, withTagAuthors } from '@/lib/tagAuthors';

// tagsApi.getAll — list tags. Viewers too, unlike the rest of this file: the
// tag picker on every detail page offers the whole vocabulary, and a tag's name
// and id say nothing about the catalog.
//
// Author names are added only when asked for — see `authorsRequested`. This is
// the hottest tag route there is, read on every navigation the picker survives,
// and the settings page is the only caller with anything to show them in.
export const GET = withPermission({ tag: ['read'] })(async (req, { user }) =>
	authorsRequested(req, user) ? withTagAuthors(await proxyToBackend(req)) : proxyToBackend(req),
);

// tagsApi.create — add a tag (admin only). The identity goes down so the
// backend can record who created it, and comes back resolved to a name
// unconditionally: this answer only ever redraws the row on the settings page,
// which is the page that shows authors.
export const POST = withPermission({ tag: ['manage'] })(async (req, { user }) =>
	withTagAuthors(await proxyToBackend(req, { userId: user.id })),
);
