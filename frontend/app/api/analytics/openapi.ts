// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { z } from 'zod';
import type { OpenApiRoute } from '@/types/openapi';

// The row carries only `user_id`; the handler joins the User so the report can
// show a name without a second request.
// Exported so `lib/apiContract.ts` can prove it matches the Prisma select.
export const analyticsRow = z.object({
	id: z.string().meta({ format: 'uuid' }),
	user_id: z.string(),
	source: z.string().describe('`app` for the web UI, `api` for direct/API callers.'),
	question: z.string(),
	question_timestamp: z.string().meta({ format: 'date-time' }),
	response: z.string().nullable().describe('Null while the answer is still streaming.'),
	response_timestamp: z.string().meta({ format: 'date-time' }).nullable(),
	sql: z.string().nullable(),
	user: z.object({
		id: z.string(),
		name: z.string(),
		email: z.string(),
		role: z.string().describe('`admin` or `viewer`; `viewer` when the column is null.'),
	}),
});

export const openapi: OpenApiRoute = {
	get: {
		query: z.object({
			skip: z.int().optional().describe('Rows to skip. Unparseable values fall back to 0.'),
			limit: z
				.int()
				.optional()
				.describe('Page size. Omitted returns every row in the window.'),
		}),
		responses: {
			200: {
				description:
					'One page of the trailing 30-day question log, newest first. `total` ' +
					'counts the whole window, not the page.',
				schema: z.object({ data: z.array(analyticsRow), total: z.int() }),
			},
		},
	},
};
