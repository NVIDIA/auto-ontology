// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** Cap on `POST /search/global-search`. Count is uncapped. */
export const GLOBAL_SEARCH_LIST_LIMIT = 200;

/**
 * Same floor as the backend `MIN_SEARCH_LENGTH`.
 * Two-character queries cannot use `gin_trgm_ops`; we still allow them
 * (`id`, `BU`) and rely on the 1000 ms debounce instead of raising the floor.
 */
export const GLOBAL_SEARCH_MIN_QUERY_LENGTH = 2;

/** Tab id for the unfiltered global-search list. */
export const GLOBAL_SEARCH_ALL_TAB = 'all';

/**
 * The value `filters.tags` carries to mean "objects with no tags at all".
 *
 * Mirrors `UNTAGGED_FILTER_VALUE` in `auto_ontology/server/search/constants.py`,
 * which is what reads it. A sentinel inside the list rather than a flag beside
 * it because that is what the control is: one more option in the tag picker,
 * tickable alongside real tags. Parenthesised so it cannot collide with a tag
 * id, which is a UUID.
 *
 * A rule may not carry it — the rules route answers 400 — because a rule that
 * tags what has no tags stops matching what it just labelled.
 */
export const UNTAGGED_TAG_FILTER = '(blanks)';
