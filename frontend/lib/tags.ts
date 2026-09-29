// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * The tags-on-an-object half of tagging, shared by every page that has a
 * detail view: terms, their column and SQL attributes, and catalog tables and
 * columns.
 *
 * The three pieces here are the three a page needs, and they are together
 * because each one is meaningless without the others agreeing on `tags` as the
 * section id: the page reads the options, builds the section, and — once Save
 * hands the staged tag ids back — writes the difference.
 */

import type { QueryClient } from '@tanstack/react-query';

import { tagsApi } from '@/api/tags';
import { tagQueries } from '@/lib/queries/tags';
import { ComposerSectionKind } from '@/enums/datasources';
import type { TagItemType } from '@/enums/tags';
import type { ComposerEntityTagsSection } from '@/types/composer-section';
import type { TagChip } from '@/types/tags';

/**
 * Section id the tags card stages its edit under, and so the key
 * `onPatchEdits` finds the intended tag ids at. Shared because the two halves
 * of one contract — the section a page builds and the save handler that reads
 * the edit — are written in different files.
 */
export const TAGS_SECTION_ID = 'tags';

/**
 * Every tag that exists, which is what the picker offers.
 *
 * A failed read answers `[]`: the chips an object already carries come from its
 * own payload and stay readable, so an empty picker (which explains itself) is
 * a smaller loss than failing the page over a control. Call it beside the
 * object's own fetch — the section is then built from one snapshot.
 *
 * Read through the query cache rather than straight from the API, which is
 * what stops every detail page open from re-reading the whole vocabulary: the
 * root layout has already started that read, and a copy still inside its
 * `staleTime` is handed back without asking again.
 *
 * `fetchQuery` rather than `ensureQueryData`, which returns whatever is held
 * however old it is. That is not enough here, because nothing on a detail
 * page subscribes to the vocabulary: this reads it without an observer, so
 * the invalidation the settings page fires after a rename finds no active
 * query to refetch and only marks the entry. `fetchQuery` reads that mark —
 * an invalidated query is stale whatever its age — and so re-reads before
 * answering, where `ensureQueryData` would go on offering the old name.
 *
 * `retry` is restated because `fetchQuery` turns it off where the caller
 * leaves it unset, and the one retry the provider asks for is wanted here
 * too: this answers `[]` on failure, and a dropped connection emptying the
 * picker is worth a second attempt.
 *
 * Takes the client rather than reaching for it, because a page builds its
 * sections inside a callback rather than while rendering, and hooks cannot be
 * called from there.
 */
export const fetchTagOptions = async (queryClient: QueryClient): Promise<TagChip[]> => {
	try {
		return await queryClient.fetchQuery({ ...tagQueries.vocabulary(), retry: 1 });
	} catch {
		return [];
	}
};

/** The tags section as every detail page wants it: editable, titled, in place. */
export const entityTagsSection = (
	tags: TagChip[] | null | undefined,
	options: TagChip[],
): ComposerEntityTagsSection => ({
	type: ComposerSectionKind.ENTITY_TAGS,
	id: TAGS_SECTION_ID,
	title: 'Tags',
	tags: tags ?? [],
	options,
	editable: true,
});

/** Whether Save was handed a tags edit, and what it asked for. */
export const stagedTagIds = (edit: string | string[] | undefined): string[] | null =>
	Array.isArray(edit) ? edit : null;

export type TagSyncResult = {
	/** Null when every write landed. */
	error: string | null;
	/**
	 * The object's tags as the server last described them, for a caller that
	 * redraws from state rather than refetching. After a failure this is
	 * whatever the writes got to before it, which is what the object now
	 * carries.
	 */
	tags: TagChip[];
};

/**
 * Brings the object's tags to *nextIds*, the set a tags section staged.
 *
 * Only the difference is written, since attach and detach each take one tag,
 * and they run one after another rather than at once: every response is the
 * object's *whole* tag set afterwards, so serialising them is what makes the
 * last one a description of the finished state rather than of a race.
 */
export const syncTags = async ({
	type,
	itemId,
	current,
	nextIds,
}: {
	type: TagItemType;
	itemId: string;
	current: TagChip[];
	nextIds: string[];
}): Promise<TagSyncResult> => {
	const held = new Set(current.map((tag) => tag.id));
	const wanted = new Set(nextIds);
	const writes = [
		...nextIds
			.filter((id) => !held.has(id))
			.map((id) => () => tagsApi.attach(id, { type, id: itemId })),
		...[...held]
			.filter((id) => !wanted.has(id))
			.map((id) => () => tagsApi.detach(id, type, itemId)),
	];

	let tags = current;
	for (const write of writes) {
		const res = await write();
		if (res.error) {
			return { error: res.message ?? 'Failed to update tags', tags };
		}
		tags = res.data ?? tags;
	}
	return { error: null, tags };
};
