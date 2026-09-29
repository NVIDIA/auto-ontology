// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Resolves the Better Auth user ids a tag read carries to the names the Tags
 * settings pages render: a tag's own `created_by` / `modified_by`, and the
 * `tagged_by` on each object a tag labels, which is a page of its own.
 *
 * Here, in the gateway, for two reasons. FastAPI cannot do it: the accounts live
 * in the `frontend` schema Prisma owns, which the backend has no model for and
 * deliberately does not reach into. And the browser should not: the admin
 * `listUsers` call is a second round trip returning a *page* of users, so every
 * author past that page would render as "Auto Generated" — which is also what a
 * deleted account renders as, making the two indistinguishable.
 *
 * The same join as the analytics report (`lib/apiSelects.ts`), for the same
 * reason: the row stores an id, and only the User table knows the name.
 */

import { getPrisma } from '@/lib/prisma';
import { userCan } from '@/auth/permissions';
import { PROXY_CACHE_CONTROL } from '@/auth/proxy-backend';
import type { ResolvedUser } from '@/auth/resolve-user';
import { AUTHORS_PARAM, AUTHORS_PARAM_ON } from '@/constants/tags';

/**
 * Whether this read should carry authors: the caller asked for them, and may
 * have them.
 *
 * Both halves are needed and neither implies the other. The param is what keeps
 * the work off the tag picker's path, which reads the same list on every detail
 * page and throws authors away. The permission is what stops a viewer from
 * helping themselves to the names by appending the param — only the settings
 * page shows who curated the vocabulary, and only admins can open it.
 *
 * One function rather than a check per route, so the two read routes cannot
 * come to disagree about who sees authors.
 */
export const authorsRequested = (req: Request, user: ResolvedUser): boolean =>
	new URL(req.url).searchParams.get(AUTHORS_PARAM) === AUTHORS_PARAM_ON &&
	userCan(user, { tag: ['manage'] });

/** Only what a row needs: a name to print, and an email to fall back to. */
const authorSelect = { id: true, name: true, email: true } as const;

type Row = Record<string, unknown>;

const isRow = (value: unknown): value is Row => typeof value === 'object' && value !== null;

/**
 * The rows in a `{ data }` envelope — one from a write, a list from a read.
 *
 * Tags on the tag routes, and the objects a tag labels on its `targets` route.
 * The two are read the same way here because they are enriched the same way:
 * whichever author columns a row carries get resolved, and a row carrying none
 * costs nothing.
 */
const rowsIn = (payload: Row): Row[] => {
	const rows = Array.isArray(payload.data) ? payload.data : [payload.data];
	return rows.filter(isRow);
};

/**
 * The id to look up, or null when the column holds nothing to look one up by.
 *
 * An id that names no account is not filtered out here, and does not need to
 * be: the lookup simply does not find it, which the pages render exactly as
 * they render a null — "Auto Generated". That covers a deleted account and the
 * legacy `system` an earlier version stored alike.
 */
const lookupId = (value: unknown): string | null =>
	typeof value === 'string' && value !== '' ? value : null;

/**
 * The upstream answer, rebuilt around a body this module may have rewritten.
 *
 * The cache policy is *carried* rather than decided again. This wrapper sits on
 * top of `proxyToBackend`, which is where a proxied answer's policy is set, and
 * rebuilding a response is exactly where such a header goes missing — the enriched
 * branch of `/api/tags` would then be the one reply on that route with no policy
 * at all, while carrying the most caller-specific body of the two.
 */
const respond = (body: string, upstream: Response): Response =>
	new Response(body, {
		status: upstream.status,
		headers: {
			'Content-Type': upstream.headers.get('content-type') ?? 'application/json',
			'Cache-Control': upstream.headers.get('cache-control') ?? PROXY_CACHE_CONTROL,
		},
	});

/**
 * The backend's answer with `created_by_user` / `modified_by_user` — and, on a
 * page of tagged objects, `tagged_by_user` — added beside the ids it stored.
 *
 * Every branch that cannot add them answers with the body unchanged rather than
 * failing: an error response, a body that is not the expected envelope, or a
 * failed user lookup. A tag whose author will not resolve still reads as "Auto
 * Generated", whereas a 500 here would lose the whole list over a decoration.
 */
export const withTagAuthors = async (upstream: Response): Promise<Response> => {
	const body = await upstream.text();
	if (!upstream.ok) return respond(body, upstream);

	let payload: unknown;
	try {
		payload = JSON.parse(body);
	} catch {
		return respond(body, upstream);
	}
	if (typeof payload !== 'object' || payload === null) return respond(body, upstream);

	// Mutated in place below, so the envelope keeps whatever else it carries —
	// a list's `count`, a page's `total`.
	const rows = rowsIn(payload as Row);
	// One lookup for all three columns, since a tag curated and a label applied
	// are commonly the same person and the union is what makes that one row read.
	const ids = [
		...new Set(
			rows
				.flatMap((row) => [
					lookupId(row.created_by),
					lookupId(row.modified_by),
					lookupId(row.tagged_by),
				])
				.filter((id): id is string => id !== null),
		),
	];
	if (ids.length === 0) return respond(body, upstream);

	let authors: Map<string, unknown>;
	try {
		const users = await getPrisma().user.findMany({
			where: { id: { in: ids } },
			select: authorSelect,
		});
		authors = new Map(users.map((user) => [user.id, user]));
	} catch {
		return respond(body, upstream);
	}

	// Set only where the row has the column, so a tag does not gain a
	// `tagged_by_user` and a tagged object does not gain an author it has no id
	// for — a null there means "no account", which is a claim about a column
	// this row does not carry.
	for (const row of rows) {
		if ('created_by' in row || 'modified_by' in row) {
			const createdBy = lookupId(row.created_by);
			const modifiedBy = lookupId(row.modified_by);
			row.created_by_user = createdBy === null ? null : (authors.get(createdBy) ?? null);
			row.modified_by_user = modifiedBy === null ? null : (authors.get(modifiedBy) ?? null);
		}
		if ('tagged_by' in row) {
			const taggedBy = lookupId(row.tagged_by);
			row.tagged_by_user = taggedBy === null ? null : (authors.get(taggedBy) ?? null);
		}
	}

	return respond(JSON.stringify(payload), upstream);
};
