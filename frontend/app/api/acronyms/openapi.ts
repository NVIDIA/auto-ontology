// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { z } from 'zod';
import { acronymSchema } from '@/lib/openapiSchemas';
import type { OpenApiRoute } from '@/types/openapi';

export const openapi: OpenApiRoute = {
	get: {
		query: z.object({
			name: z
				.string()
				.optional()
				.describe('When given, the route answers the existence check instead of listing.'),
		}),
		responses: {
			200: {
				description:
					'Every acronym, newest first — or `{ "exists": true|false }` when `name` ' +
					'is given, which the create form uses to check the unique constraint.',
				schema: z.union([z.array(acronymSchema), z.object({ exists: z.boolean() })]),
			},
		},
	},
	post: {
		body: {
			description: '`name` is unique across the instance.',
			schema: z.object({
				name: z.string(),
				description: z.string().optional(),
			}),
		},
		responses: {
			201: { description: 'The created acronym.', schema: acronymSchema },
			500: {
				description:
					'`name` is already taken (the unique-constraint error is not mapped to a 409).',
			},
		},
	},
};
