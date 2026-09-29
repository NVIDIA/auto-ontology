// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { requests } from './requests';
import type { Connection } from '@/types/connection';
import type { ConnectionInput } from '@/types/connection';
import type { ApiError, ApiResponse, ResponseWithCount } from './types';

type CreateResponse = ApiError | { data: Connection };
// A successful test also returns the connection's schemas (for the picker).
type TestResponse = ApiError | { success: true; schemas: string[] };

const err = (message: string): ApiError => ({ error: true, message });

export const connectionsApi = {
	getAll: (): Promise<ApiResponse<Connection[]>> =>
		requests.get<ResponseWithCount<Connection[]>>('connections'),

	isEnvSource: () => requests.get<boolean>('connections/source'),

	create: (input: ConnectionInput): Promise<CreateResponse> =>
		requests.post<{ data: Connection }>('connections', { connection: input }),

	test: async (input: ConnectionInput, replacing?: string): Promise<TestResponse> => {
		const res = await requests.post<{ success: boolean; schemas?: string[] }>(
			'connections/test',
			{ connection: input, ...(replacing == null ? {} : { replacing }) },
		);

		if (res.error || res.success !== true) {
			return err(res.message ?? 'Connection test failed.');
		}

		// The test also reports the connection's schemas (empty for connectors
		// that don't enumerate them) so the UI can offer a schema picker.
		return { success: true, schemas: res.schemas ?? [] };
	},

	// Rewrites every setting except the connection's identity: `database` (which
	// doubles as the connection's name) and `type` come back as a 422 if they
	// differ from the stored ones. Credentials left blank keep their stored
	// value, since `getAll` strips them and the edit form has none to send back.
	update: (databaseName: string, input: ConnectionInput): Promise<CreateResponse> =>
		requests.put<{ data: Connection }>(`connections/${encodeURIComponent(databaseName)}`, {
			connection: input,
		}),

	// Narrow update: flips "authenticate as signed-in user" without re-sending
	// credentials, so the connections page can toggle it inline.
	setSsoFederation: (databaseName: string, enabled: boolean) =>
		requests.patch<{ data: { database_name: string; sso_federation: boolean } }>(
			`connections/${encodeURIComponent(databaseName)}/sso-federation`,
			{ enabled },
		),

	delete: (databaseName: string) =>
		requests.delete<{ data: { database_name: string } }>(
			`connections/${encodeURIComponent(databaseName)}`,
		),
};
