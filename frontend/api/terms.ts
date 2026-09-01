// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { pageQuery, requests } from './requests';
import type { CertificationStatus } from '@/enums/certification';
import type {
	ColumnAttribute,
	RelatedTermCount,
	SqlAttribute,
	Term,
	TermCount,
	TermDetail,
} from '@/types/terms';
import type { PageParams, ResponseWithError } from './types';

/** One page of a list endpoint: `count` is this page, `total` the whole list. */
type PagedResult<T> = { data: T[]; count: number; total: number };

type AttributeListResult = PagedResult<ColumnAttribute>;
type AttributeListResponse = ResponseWithError<AttributeListResult>;

type SqlAttributeListResult = PagedResult<SqlAttribute>;
type SqlAttributeListResponse = ResponseWithError<SqlAttributeListResult>;

type SingleResult = { data: TermDetail };
type SingleResponse = ResponseWithError<SingleResult>;

type TermUpdateResult = {
	data: {
		id: string;
		name: string;
		description: string | null;
		name_certified: boolean;
		description_certified: boolean;
		certification: CertificationStatus;
	};
};
type TermUpdateResponse = ResponseWithError<TermUpdateResult>;

type ColumnAttributeUpdateResult = {
	data: {
		id: string;
		name: string;
		description: string | null;
		sample_values?: string[] | null;
		certified: boolean;
	};
	/** Owning term's aggregate status, recomputed server-side after the write. */
	term_certification: CertificationStatus | null;
};
type ColumnAttributeUpdateResponse = ResponseWithError<ColumnAttributeUpdateResult>;

type TermUpdatePayload = {
	name?: string;
	description?: string | null;
	name_certified?: boolean;
	description_certified?: boolean;
};
type ColumnAttributeUpdatePayload = {
	name?: string;
	description?: string | null;
	sample_values?: string[];
	certified?: boolean;
};

type ListResult = {
	terms: Term[];
	/** Terms matching the filter in full, not the length of `terms`. */
	total: number;
	/**
	 * Per-term badge counts. When a `limit` was requested these describe only
	 * the terms on this page, so a caller accumulating pages must merge them
	 * into what it already holds instead of replacing it.
	 */
	column_attribute_counts: TermCount[];
	sql_attribute_counts: TermCount[];
	related_counts: RelatedTermCount[];
};
type ListResponse = ResponseWithError<ListResult>;

export type TermsListParams = PageParams & {
	/** Case-insensitive substring filter on the term name. */
	q?: string;
};

export const termsApi = {
	list: (params?: TermsListParams): Promise<ListResponse> =>
		requests.get<ListResult>('terms', {
			...(params?.q ? { q: params.q } : {}),
			...pageQuery(params),
		}),
	get: (id: string): Promise<SingleResponse> => requests.get<SingleResult>(`terms/${id}`),
	update: (id: string, payload: TermUpdatePayload): Promise<TermUpdateResponse> =>
		requests.patch<TermUpdateResult>(`terms/${encodeURIComponent(id)}`, payload),
	getColumnAttributes: (id: string, params?: PageParams): Promise<AttributeListResponse> =>
		requests.get<AttributeListResult>(`terms/${id}/column-attributes`, pageQuery(params)),
	updateColumnAttribute: (
		termId: string,
		attrId: string,
		payload: ColumnAttributeUpdatePayload,
	): Promise<ColumnAttributeUpdateResponse> =>
		requests.patch<ColumnAttributeUpdateResult>(
			`terms/${encodeURIComponent(termId)}/column-attributes/${encodeURIComponent(attrId)}`,
			payload,
		),
	getSqlAttributes: (id: string, params?: PageParams): Promise<SqlAttributeListResponse> =>
		requests.get<SqlAttributeListResult>(`terms/${id}/sql-attributes`, pageQuery(params)),
};
