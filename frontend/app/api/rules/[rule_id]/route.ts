// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';

// rulesApi.update — rename a rule (admin only).
//
// The identity goes with it, as it does with the create, but for the other
// column: `created_by` is left alone so the rule stays attributed to whoever
// saved it, and `modified_by` records who renamed it — which is what the
// information card reads to say who last edited the rule.
export const PATCH = withPermission({ tag: ['manage'] })((req, { user }) =>
	proxyToBackend(req, { userId: user.id }),
);

// rulesApi.delete — remove a rule, and with it the tags it applied unless
// `?keep_tags=true` says to leave them (admin only). `tag: ['manage']` rather
// than the `catalog: ['edit']` that guards applying a single tag: this un-tags
// every object the rule matched in one request, and can remove a tag from the
// vocabulary outright when the rule was the whole of what used it — a change to
// the vocabulary rather than an edit of any one object.
//
// No identity forwarded: nothing survives the delete to record an actor on.
export const DELETE = withPermission({ tag: ['manage'] })((req) => proxyToBackend(req));
