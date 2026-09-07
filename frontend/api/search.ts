// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { requests } from './requests';
import { TextMatchOption } from '@/enums/search';
import type { GlobalSearchItem, GlobalSearchRequest } from '@/types/search';
import type { ResponseWithCount, ResponseWithError } from './types';

type GlobalSearchListResponse = ResponseWithError<ResponseWithCount<GlobalSearchItem[]>>;
type GlobalSearchCountResponse = ResponseWithError<{ data: Record<string, number> }>;

/**
 * Unwrap `{ data, count }` from `/search/global-search`.
 *
 * `data` is absent only when `requests` replaced the body with an `ApiError`,
 * so callers must check `response.error` themselves to tell a failed search
 * apart from one that simply matched nothing.
 */
export const globalSearchItemsFromResponse = (
	response: GlobalSearchListResponse,
): GlobalSearchItem[] => (response.error ? [] : response.data);

/** Unwrap `{ data: { type: n } }` from `/search/global-search/count`. */
export const globalSearchCountsFromResponse = (
	response: GlobalSearchCountResponse,
): Record<string, number> => (response.error ? {} : response.data);

const withDefaults = (payload: GlobalSearchRequest): GlobalSearchRequest => ({
	text_match_option: TextMatchOption.Contains,
	...payload,
	filters: { description: true, ...payload.filters },
});

export const searchApi = {
	globalSearch: (
		payload: GlobalSearchRequest,
		abortController?: AbortController,
	): Promise<GlobalSearchListResponse> =>
		requests.post<ResponseWithCount<GlobalSearchItem[]>>(
			'search/global-search',
			withDefaults(payload),
			abortController,
		),
	globalSearchCount: (
		payload: GlobalSearchRequest,
		abortController?: AbortController,
	): Promise<GlobalSearchCountResponse> =>
		requests.post<{ data: Record<string, number> }>(
			'search/global-search/count',
			withDefaults(payload),
			abortController,
		),
};
