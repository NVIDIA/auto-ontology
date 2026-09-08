// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Where a tagged object's own page lives, per kind.
 *
 * Its own module rather than part of `tags-path.ts`: that file is the Tags
 * screen's own URL, which the list and the detail view both need, while this
 * one points *away* from it into Terms and the data catalog.
 */

import { TagItemType } from '@/enums/tags';
import type { TagItem } from '@/types/tags';

const focusUrl = (page: string, focusId: string): string =>
	`${page}?focus=${encodeURIComponent(focusId)}`;

/**
 * A catalog node, which the data page addresses by its whole chain of ids in
 * one `focus` value — `dbId|schemaId|tableId|columnId`, truncated to the depth
 * of the node.
 *
 * A gap anywhere in the chain makes the URL unbuildable rather than shorter: a
 * shorter one is a valid address for an *ancestor*, so falling back to it would
 * silently open the wrong page.
 */
const catalogPath = (...ids: (string | null)[]): string | null =>
	ids.some((id) => id == null || id === '') ? null : focusUrl('/data', ids.join('|'));

/**
 * An attribute, whose page is a section of its term's and so needs both ids:
 * the term in `focus`, the attribute in the param that says which kind it is.
 */
const attributePath = (param: string, termId: string | null, attrId: string): string | null =>
	termId == null || termId === ''
		? null
		: `${focusUrl('/terms', termId)}&${param}=${encodeURIComponent(attrId)}`;

/**
 * The page for one tagged object, or null when there is nowhere to send.
 *
 * Null should not occur: an attribute is always a property of one term (the
 * backend refuses to tag one that is not), and a catalog object always has its
 * whole chain. It is still a case rather than a cast, because the alternative
 * to returning null is guessing a URL, and the row simply staying inert is the
 * cheaper way to be wrong.
 */
export const tagItemPath = (item: TagItem): string | null => {
	switch (item.type) {
		case TagItemType.Term:
			return focusUrl('/terms', item.id);
		case TagItemType.Table:
			return catalogPath(item.database_id, item.schema_id, item.id);
		case TagItemType.Column:
			return catalogPath(item.database_id, item.schema_id, item.table_id, item.id);
		case TagItemType.ColumnAttribute:
			return attributePath('colAttr', item.term_id, item.id);
		case TagItemType.SqlAttribute:
			return attributePath('sqlAttr', item.term_id, item.id);
	}
};
