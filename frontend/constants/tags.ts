// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Query param asking `/api/tags` to resolve `created_by` / `modified_by` to the
 * accounts they name.
 *
 * Opt-in rather than given to everyone who could see it. The same list is read
 * by the tag picker on every catalog and glossary detail page — on each
 * navigation, and again after each save — and the picker wants a name and an id
 * to put on a chip. Resolving authors for it would be a user lookup per page
 * open whose result nothing reads.
 */
export const AUTHORS_PARAM = 'authors';

/** The value {@link AUTHORS_PARAM} is asked with. */
export const AUTHORS_PARAM_ON = '1';

/**
 * The `created_by` / `modified_by` a tag carries when no person asked for it.
 * Mirrors `SYSTEM_ACTOR` in `gsf/dal/tags.py`.
 *
 * Distinct from a null, which means the author was never recorded — a tag made
 * by a caller that reached the backend without the gateway. Both are rendered,
 * and differently: one deployment made the tag, the other simply cannot say who
 * did.
 */
export const SYSTEM_ACTOR = 'system';

/** Shown for `SYSTEM_ACTOR`. */
export const SYSTEM_ACTOR_LABEL = 'Auto Generated';

/**
 * Shown for a null author, and for an id whose account is gone — the ids are
 * not foreign keys, so a deleted user leaves one behind that resolves to
 * nothing. Both are the same thing to a reader: there is a tag, and nobody left
 * to attribute it to.
 */
export const UNKNOWN_ACTOR_LABEL = 'Unknown';
