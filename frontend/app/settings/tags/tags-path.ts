// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * The URL of the Tags settings screen, and of one tag within it.
 *
 * Its own module so the list and the detail view can each import it without
 * importing each other — the list renders the detail, so a constant living in
 * either file would make the pair a cycle.
 */

export const TAGS_PATH = '/settings/tags';

/** Query param naming the open tag. Same name the Terms and catalog pages use. */
export const FOCUS_PARAM = 'focus';

/** A selected tag, as `/settings/tags?focus=<id>`. */
export const tagPath = (tagId: string): string =>
	`${TAGS_PATH}?${FOCUS_PARAM}=${encodeURIComponent(tagId)}`;

/** The settings panel's own frame, shared by the list and the detail view. */
export const TAGS_PANEL_CLASSNAME =
	'min-h-0 min-w-0 flex-1 overflow-y-auto bg-[linear-gradient(180deg,rgba(255,255,255,1)_0%,rgba(250,250,250,0.6)_100%)] px-7 py-6 sm:px-10 sm:py-7 dark:bg-[linear-gradient(180deg,rgba(9,9,11,1)_0%,rgba(24,24,27,0.5)_100%)]';
