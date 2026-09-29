// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * The two ways global search splits a query, mirroring `auto_ontology/dal/search.py`.
 *
 * The backend reads one query twice — as substrings for name and description,
 * as whole words for synonyms — and the two splits are not the same. Anything
 * that shows the user *why* a row matched has to pick the one the match was
 * made with, or it highlights text that had nothing to do with it.
 */

/**
 * Token separators, mirroring `_SEPARATORS`.
 *
 * `_` and `.` are absent on purpose: the backend matches `total_amount` and
 * `orders.id` as one token each, so splitting them here would mark `total` in
 * a description that never contained `total_amount`.
 */
const SEARCH_SEPARATORS = /[-\\/#%*,"$&?!@^<>|+:;~(){}[\]]/g;

/**
 * The tokens a name or description contains, as `search_tokens` splits them.
 *
 * Deduplicated, which the backend does not need to be: repeating a token only
 * ANDs the same condition twice there, while here it would build an alternation
 * with the same branch twice.
 */
export const searchTokens = (query: string): string[] => [
	...new Set(
		query
			.toLowerCase()
			.replace(SEARCH_SEPARATORS, ' ')
			.split(/\s+/)
			.filter((token) => token !== ''),
	),
];

/** The whole words a synonym must contain, as `synonym_word_tokens` splits them. */
export const synonymWordTokens = (query: string): string[] => [
	...new Set(query.toLowerCase().match(/[a-z0-9]+/g) ?? []),
];
