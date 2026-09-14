// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';
import { withTagAuthors } from '@/lib/tagAuthors';

// rulesApi.getAll — list rules.
//
// `manage` rather than the `read` that guards the tag *list*: a rule names the
// search it replays, so the list of them describes the catalog the way a tag's
// own page does. The only screen that asks for it is Settings → Rules, and that
// whole section is admin-only already.
//
// `withTagAuthors` despite the name: it resolves `created_by` / `modified_by` in
// a `{ data }` envelope, and a rule stores those two columns for the same reason
// a tag does — the accounts live in the `frontend` schema, which the backend has
// no model for. Unconditionally, without the `?authors=` the tag list gates it
// behind: that param exists because the tag picker reads the vocabulary on every
// navigation and throws the names away, and nothing reads rules but the one page
// that shows an owner on every card.
export const GET = withPermission({ tag: ['manage'] })(async (req) =>
	withTagAuthors(await proxyToBackend(req)),
);

// rulesApi.create — save a rule (admin only).
//
// Guarded like creating a tag rather than like applying one: a rule labels
// everything its search matches, and goes on labelling whatever matches later,
// so it curates the vocabulary's reach rather than putting one existing tag on
// one object.
//
// The identity goes with it because the backend attributes the rule to whoever
// saved it — `created_by` — and `proxyToBackend` forwards nothing from the
// incoming request. `ctx.user` is the resolved caller, so the id the backend
// trusts is the one this route authenticated.
export const POST = withPermission({ tag: ['manage'] })((req, { user }) =>
	proxyToBackend(req, { userId: user.id }),
);
