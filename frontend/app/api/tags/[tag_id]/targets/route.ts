// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';
import { withTagAuthors } from '@/lib/tagAuthors';

// tagsApi.getTargets — one page of the objects a tag labels.
//
// `tag: ['manage']`, not the `catalog: ['edit']` below: this answer names
// catalog objects and spells out where they sit, and it is not scoped to the
// caller's zones the way the catalog reads are. Only the tag's settings page
// asks for it, and that section is admin-only already.
//
// Authors unconditionally, without the `?authors=` the tag list gates them
// behind: the "Tagged By" column is a column of this table rather than an extra
// a caller may not render, so a page of rows with unresolved ids would be a
// page with a blank column.
export const GET = withPermission({ tag: ['manage'] })(async (req) =>
	withTagAuthors(await proxyToBackend(req)),
);

// tagsApi.attach — label an object with an existing tag.
//
// `catalog: ['edit']` rather than `tag: ['manage']`, which is what guards the
// tag list itself: curating the vocabulary is an admin's job, but applying a
// tag that already exists is an edit of the object, no different from editing
// its description. Viewers hold that permission and edit terms with it.
//
// The identity goes down for the backend's `tagged_by`, which the tag's page
// shows in its "Tagged By" column — the one thing distinguishing a label a
// person applied from one a rule matched.
export const POST = withPermission({ catalog: ['edit'] })(async (req, { user }) =>
	proxyToBackend(req, { userId: user.id }),
);
