// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { SearchObjectType } from '@/enums/search';
import type { GlobalSearchRequest } from '@/types/search';
import type { TagAuthor, TagChip } from '@/types/tags';

/**
 * A rule as its dialog has it, before anything is saved.
 *
 * A rule labels whatever a search matches — now and later — so what identifies
 * it is the tags to apply; the search itself is held by the screen the dialog
 * was opened from, which is what turns this into a `RuleCreateInput`.
 */
export type RuleTagDraft = {
	name: string;
	/** Applied to every matched item. */
	tags: TagChip[];
};

/**
 * A rule as it is saved.
 *
 * Built on `GlobalSearchRequest` rather than restating its three fields: a rule
 * *is* a saved search, and the request it replays has to be the request the
 * search accepts or the rule can never reproduce what it was created from.
 */
export type RuleCreateInput = GlobalSearchRequest & {
	name: string;
	/**
	 * The chips the dialog holds. Only `id` is read: everything else about a tag
	 * is the tag table's, and a rule answers with each tag as it is now — so a
	 * renamed tag reads back renamed on every rule applying it.
	 */
	tags: TagChip[];
};

/**
 * A rule edit, which is a rename and nothing else.
 *
 * Not the search and not the tags: those are what the rule *is*, and changing
 * either would move which objects it labels while leaving the labels it already
 * wrote in place. A name only says how this list refers to the rule — the
 * labels name it by id, so a rename reaches every one of them.
 *
 * Unique across rules, folded for case and surrounding space, as a tag name is:
 * a name that named two rules would leave a reader unable to say which one they
 * are deleting, and deleting takes back everything that rule labelled.
 */
export type RuleUpdateInput = {
	name: string;
};

/**
 * The search filters a rule replays.
 *
 * Every field is present here, unlike the `GlobalSearchFilters` a request
 * sends: a stored rule has resolved the defaults, so there is no longer an
 * omitted flag to interpret.
 *
 * `objects` is null for a rule saved from the search's All tab, which narrows
 * to no particular kind.
 */
export type RuleFilters = {
	description: boolean;
	synonyms: boolean;
	objects: SearchObjectType[] | null;
};

/** A stored rule, as every read returns it. */
export type Rule = {
	id: string;
	name: string;
	search_term: string;
	text_match_option: string;
	filters: RuleFilters;
	/** Applied to every item the search matches. */
	tags: TagChip[];
	/** Id of the user who saved it. */
	created_by: string;
	/**
	 * The account that id names, added by the `/api/rules` route — the same join
	 * the tag list does, since the ids are stored in the catalog and the accounts
	 * live in the `frontend` schema.
	 *
	 * Null when the id names nobody: the account has since been deleted, which
	 * these ids outlive because they are not foreign keys.
	 */
	created_by_user?: TagAuthor | null;
	/**
	 * Id of the user who last renamed it, and null until somebody has — the same
	 * fact as `modified` still equalling `created`.
	 */
	modified_by: string | null;
	/** The account `modified_by` names, joined as `created_by_user` is. */
	modified_by_user?: TagAuthor | null;
	/** ISO 8601, from the backend clock. */
	created: string;
	/** ISO 8601. Equal to `created` until something edits the rule. */
	modified: string;
};
