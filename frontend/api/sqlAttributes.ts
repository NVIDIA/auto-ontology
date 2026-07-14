// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { requests } from './requests';
import type { SqlAttribute } from '@/types/terms';
import type { ResponseWithError } from './types';

type SingleResult = { data: SqlAttribute };
type SingleResponse = ResponseWithError<SingleResult>;

export type SqlAttributeCreatePayload = {
	name: string;
	description: string;
	expression: string;
	term_id: string;
	source?: string;
};

export type SqlAttributeValidatePayload = {
	expression: string;
	term_id?: string;
	attribute_id?: string;
};

export type SqlAttributePatchPayload = {
	name?: string;
	description?: string | null;
};

type ValidateResult = { data: { valid: boolean; expression: string } };
type ValidateResponse = ResponseWithError<ValidateResult>;
type CreateResponse = ResponseWithError<SingleResult>;
type UpdateResponse = ResponseWithError<SingleResult>;
type DeleteResponse = ResponseWithError<{ data: { id: string } }>;
type SuggestDescriptionResult = { data: string | null };
type SuggestDescriptionResponse = ResponseWithError<SuggestDescriptionResult>;

export const sqlAttributesApi = {
	get: (id: string): Promise<SingleResponse> =>
		requests.get<SingleResult>(`sql-attributes/${id}`),
	validate: (payload: SqlAttributeValidatePayload): Promise<ValidateResponse> =>
		requests.post<ValidateResult>('sql-attributes/validate', payload),
	create: (payload: SqlAttributeCreatePayload): Promise<CreateResponse> =>
		requests.post<SingleResult>('sql-attributes', payload),
	update: (id: string, payload: SqlAttributeCreatePayload): Promise<UpdateResponse> =>
		requests.put<SingleResult>(`sql-attributes/${encodeURIComponent(id)}`, payload),
	patch: (id: string, payload: SqlAttributePatchPayload): Promise<UpdateResponse> =>
		requests.patch<SingleResult>(`sql-attributes/${encodeURIComponent(id)}`, payload),
	delete: (id: string): Promise<DeleteResponse> =>
		requests.delete<{ data: { id: string } }>(`sql-attributes/${encodeURIComponent(id)}`),
	suggestDescription: (id: string): Promise<SuggestDescriptionResponse> =>
		requests.get<SuggestDescriptionResult>(
			`sql-attributes/${encodeURIComponent(id)}/description-suggestion`,
		),
};
