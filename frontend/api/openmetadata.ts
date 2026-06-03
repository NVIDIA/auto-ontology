// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Browser-side typed client for the OpenMetadata REST API.
 *
 * All calls go through the Next.js proxy at /api/openmetadata/<path>, which
 * injects the bot JWT on the server. The proxy preserves the standard OM
 * routing under /api/v1, so a call like `om.tables.list()` resolves to
 *   /api/openmetadata/tables       -> proxied to
 *   {OPENMETADATA_HOST}/api/v1/tables
 */

import type {
	OmClassification,
	OmLineageResponse,
	OmListResponse,
	OmQuery,
	OmTable,
	OmTagDef,
} from '@/types/openmetadata';

const BASE = '/api/openmetadata';

const qs = (params: Record<string, string | number | boolean | undefined | null>): string => {
	const entries = Object.entries(params).filter(
		([, v]) => v !== undefined && v !== null && v !== '',
	);
	if (entries.length === 0) return '';
	const sp = new URLSearchParams();
	for (const [k, v] of entries) sp.append(k, String(v));
	return `?${sp.toString()}`;
};

const json = async <T>(url: string, init?: RequestInit): Promise<T> => {
	const res = await fetch(url, {
		...init,
		headers: {
			Accept: 'application/json',
			...(init?.headers ?? {}),
		},
		cache: 'no-store',
	});
	if (!res.ok) {
		let detail = res.statusText;
		try {
			const data = (await res.json()) as { detail?: string; message?: string };
			detail = data.detail ?? data.message ?? detail;
		} catch {
			/* body wasn't JSON */
		}
		throw new Error(`OpenMetadata ${res.status}: ${detail}`);
	}
	return (await res.json()) as T;
};

export const om = {
	tables: {
		list: (
			params: {
				database?: string;
				databaseSchema?: string;
				service?: string;
				fields?: string;
				limit?: number;
				after?: string;
				before?: string;
			} = {},
		) => json<OmListResponse<OmTable>>(`${BASE}/tables${qs(params)}`),

		getByFqn: (fqn: string, fields = 'columns,tags,description,owners,usageSummary') =>
			json<OmTable>(`${BASE}/tables/name/${encodeURIComponent(fqn)}${qs({ fields })}`),

		getById: (id: string, fields = 'columns,tags,description,owners,usageSummary') =>
			json<OmTable>(`${BASE}/tables/${id}${qs({ fields })}`),

		queriesForTable: (id: string, limit = 20) =>
			json<OmListResponse<OmQuery>>(`${BASE}/tables/${id}/queries${qs({ limit })}`),

		patch: (id: string, ops: ReadonlyArray<Record<string, unknown>>) =>
			fetch(`${BASE}/tables/${id}`, {
				method: 'PATCH',
				headers: { 'Content-Type': 'application/json-patch+json' },
				body: JSON.stringify(ops),
				cache: 'no-store',
			}).then(async (r) => {
				if (!r.ok) throw new Error(`PATCH /tables/${id} -> ${r.status} ${await r.text()}`);
				return r.json() as Promise<OmTable>;
			}),
	},

	classifications: {
		list: (params: { fields?: string; limit?: number } = {}) =>
			json<OmListResponse<OmClassification>>(`${BASE}/classifications${qs(params)}`),

		create: (body: { name: string; description: string; mutuallyExclusive?: boolean }) =>
			json<OmClassification>(`${BASE}/classifications`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(body),
			}),
	},

	tags: {
		list: (params: { parent?: string; fields?: string; limit?: number } = {}) =>
			json<OmListResponse<OmTagDef>>(`${BASE}/tags${qs(params)}`),

		create: (body: { name: string; description: string; classification: string }) =>
			json<OmTagDef>(`${BASE}/tags`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(body),
			}),
	},

	queries: {
		list: (
			params: {
				fields?: string;
				limit?: number;
				entityId?: string;
				service?: string;
			} = {},
		) =>
			json<OmListResponse<OmQuery>>(
				`${BASE}/queries${qs({
					fields: 'query,users,queryDate,queryUsedIn,description,duration',
					...params,
				})}`,
			),

		patch: (id: string, ops: ReadonlyArray<Record<string, unknown>>) =>
			fetch(`${BASE}/queries/${id}`, {
				method: 'PATCH',
				headers: { 'Content-Type': 'application/json-patch+json' },
				body: JSON.stringify(ops),
				cache: 'no-store',
			}).then(async (r) => {
				if (!r.ok) throw new Error(`PATCH /queries/${id} -> ${r.status} ${await r.text()}`);
				return r.json() as Promise<OmQuery>;
			}),
	},

	lineage: {
		/**
		 * Directed table→table lineage from OpenMetadata. Returns the focused
		 * entity plus upstream/downstream nodes and edges. Edges reference node
		 * IDs via fromEntity/toEntity.
		 */
		forTable: (fqn: string, upstreamDepth = 2, downstreamDepth = 2) =>
			json<OmLineageResponse>(
				`${BASE}/lineage/table/name/${encodeURIComponent(fqn)}${qs({
					upstreamDepth,
					downstreamDepth,
				})}`,
			),
	},
};

export type Om = typeof om;
