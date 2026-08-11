// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { z } from 'zod';
import type { OpenApiRoute } from '@/types/openapi';

// The body is declared here rather than merged from `docs/openapi/backend.json`
// because this handler adds a field the backend does not accept:
// `conversation_id`, which it uses to persist the result to chat history
// before forwarding the rest upstream as the backend's `VisualizeRequest`.
export const openapi: OpenApiRoute = {
	post: {
		body: {
			description:
				'Step 2 of chat: the SQL and its already-executed result from step 1. ' +
				'A missing or unparseable body is tolerated — it degrades to `{ "charts": null }` ' +
				'rather than erroring, because the chart is a nice-to-have.',
			required: false,
			schema: z.object({
				question: z.string().min(1),
				sql: z.string().default(''),
				result: z
					.unknown()
					.describe(
						'The executed SQL result — the `sql_response_from_db` from the step 1 ' +
							'answer. Either a one-item list holding a JSON-records string, or a ' +
							'list of row objects.',
					),
				conversation_id: z
					.string()
					.optional()
					.describe(
						'When given and owned by the caller, the result and any chart are ' +
							'appended to that conversation as an assistant message. Ignored by ' +
							'the backend; consumed by this handler only.',
					),
			}),
		},
	},
};
