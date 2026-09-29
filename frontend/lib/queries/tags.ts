// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Where the tag vocabulary lives in the query cache.
 *
 * The key is here rather than at each call site because it is the thing that
 * makes them one read: the root layout prefetches under it, every picker
 * subscribes to it, and the settings page invalidates it. Spelling it twice
 * would quietly give the app two vocabularies.
 */

import { queryOptions, type QueryClient } from '@tanstack/react-query';

import { tagsApi } from '@/api/tags';
import type { TagChip } from '@/types/tags';

/** Under `['tags']`, so a write can invalidate the whole family by prefix. */
const VOCABULARY_KEY = ['tags', 'vocabulary'] as const;

const byName = (left: TagChip, right: TagChip): number =>
	left.name.toLowerCase().localeCompare(right.name.toLowerCase());

/**
 * Every tag that exists, reduced to what a picker needs.
 *
 * Held as chips rather than whole tags: the dates and author ids a tag also
 * carries are dead weight in a list read on every navigation. Measured over
 * 5k tags this is 384 KB of JSON to parse and hold instead of 974 KB — but
 * barely 4% off the wire, since the field names it drops are exactly what
 * compression handles best. The saving is parse time and memory, not bytes.
 *
 * Throws where `tagsApi` reports — the cache needs a rejection to record a
 * failed read, and it is what puts a picker into its error state.
 */
const readVocabulary = async (): Promise<TagChip[]> => {
	const res = await tagsApi.getAll();
	if (res.error) throw new Error(res.message ?? 'Failed to load tags.');
	return (res.data ?? []).map((tag) => ({ id: tag.id, name: tag.name })).sort(byName);
};

export const tagQueries = {
	/**
	 * Usable from both sides of the render, which is what lets the layout
	 * prefetch exactly what the browser then subscribes to: `requests` picks
	 * the backend directly on the server and the permission-gated `/api/tags`
	 * route in the browser, and that route is a plain proxy for a read that
	 * asks for no authors — so both get the same answer.
	 */
	vocabulary: () =>
		queryOptions({
			queryKey: VOCABULARY_KEY,
			queryFn: readVocabulary,
		}),
};

/**
 * Marks the vocabulary stale, so every picker holding it re-reads.
 *
 * What the tag settings page calls after a create, a rename or a delete: the
 * pickers elsewhere in the app are subscribers to this one list, and this is
 * the whole of keeping them honest.
 */
export const invalidateTagVocabulary = (queryClient: QueryClient): Promise<void> =>
	queryClient.invalidateQueries({ queryKey: VOCABULARY_KEY });

/**
 * Adds a tag created outside the settings page to the vocabulary already
 * loaded, rather than re-reading the list to learn about a tag we just made.
 *
 * Only where one is loaded, which is not a given: the rule panel offers to
 * create whatever was typed as soon as nothing matches it, and a vocabulary
 * that failed to read matches nothing. Adding to an empty list there would
 * publish the new tag as the entire vocabulary — a success, so the panel
 * would stop reporting the failure too. Re-reading says what is true.
 *
 * Cancels a read in flight first, for the reason the zone and connection
 * lists do: an answer given before the tag existed must not land after it.
 */
export const addTagToVocabulary = async (queryClient: QueryClient, tag: TagChip): Promise<void> => {
	await queryClient.cancelQueries({ queryKey: VOCABULARY_KEY });
	const held = queryClient.getQueryData<TagChip[]>(VOCABULARY_KEY);
	if (held === undefined) {
		await queryClient.invalidateQueries({ queryKey: VOCABULARY_KEY });
		return;
	}
	queryClient.setQueryData(VOCABULARY_KEY, [...held, tag].sort(byName));
};
