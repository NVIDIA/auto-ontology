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

import { tagsApi } from '@/api/tags';
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
 * object's own fetch — the section is then built from one snapshot, and a tag
 * created elsewhere appears on the next refetch rather than needing its own
 * subscription.
 *
 * That refetch is what keeps the picker current, and it is why this asks for no
 * authors and holds no cache. It runs on every detail page open and again after
 * every save, so anything the answer carries is paid for on each of them — and
 * a cached list would show a vocabulary that no longer matches the one the
 * settings page has just been edited in.
 */
export const fetchTagOptions = async (): Promise<TagChip[]> => {
	const res = await tagsApi.getAll();
	return res.error ? [] : (res.data ?? []);
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
