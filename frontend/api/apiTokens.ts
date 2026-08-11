// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { requests } from './requests';
import type { ResponseWithError } from './types';

export type ApiToken = {
	id: string;
	name: string | null;
	/** First characters of the token, e.g. `gsf_a1b2c3` — all the UI can show. */
	start: string | null;
	enabled: boolean;
	created_at: string;
	expires_at: string | null;
	last_request: string | null;
};

/** A freshly minted token: the only response that carries the secret itself. */
export type CreatedApiToken = ApiToken & { token: string };

export const apiTokensApi = {
	list: (): Promise<ResponseWithError<ApiToken[]>> => requests.get('api-tokens'),

	/** Omitting `expiresInDays` mints a non-expiring token. */
	create: (name: string, expiresInDays?: number): Promise<ResponseWithError<CreatedApiToken>> =>
		requests.post('api-tokens', { name, expires_in_days: expiresInDays }),

	revoke: (id: string): Promise<ResponseWithError<{ id: string }>> =>
		requests.delete(`api-tokens/${encodeURIComponent(id)}`),
};
