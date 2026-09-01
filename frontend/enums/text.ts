// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Typography presets for `Text`. Callers pick the role a value plays
 * on the page rather than the classes that render it, so the styling stays in
 * one place instead of being spelled out at every call site.
 */
export enum TextVariant {
	/** Takes the surrounding typography as-is. Table cells, tree rows, tags. */
	Inherit = 'inherit',
	/** The name of the entity a page is about. */
	PageTitle = 'pageTitle',
	/** The name of the entity a card is about. */
	CardTitle = 'cardTitle',
	/** Section and modal headings. */
	Heading = 'heading',
	/** A heading of secondary weight, for cards inside a list. */
	Subheading = 'subheading',
	/** Running text — descriptions, prompts, anything paragraph-shaped. */
	Body = 'body',
	/** A value emphasised against its neighbours, keeping the inherited colour. */
	Strong = 'strong',
	/** Supporting line under a heading — synonyms, an email under a name. */
	Caption = 'caption',
	/** Small print that still has to be read, such as a list item's description. */
	Detail = 'detail',
	/** Name of a field, shown above or beside its value. */
	Label = 'label',
	/** Uppercase eyebrow above a block of content. */
	Overline = 'overline',
}
