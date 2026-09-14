// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** Mirrors `MAX_TAG_NAME_LENGTH` in `gsf/server/tags/router.py`, which rejects longer. */
export const MAX_TAG_NAME_LENGTH = 25;

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
 * Shown wherever a person would be named and there is none.
 *
 * The only alternative to an account, deliberately: a tag with no author, a
 * label with no account and no rule, and an id whose account has since been
 * deleted all read the same way. A reader can act on "a person or a rule did
 * this" and can do nothing with which flavour of nobody it was, so no screen
 * offers a second answer for it.
 */
export const AUTO_GENERATED_LABEL = 'Auto Generated';
