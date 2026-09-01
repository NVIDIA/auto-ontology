// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { z } from 'zod';
import { acronymSchema } from '@/lib/openapiSchemas';
import type { OpenApiRoute } from '@/types/openapi';

// Both methods call Prisma directly, which raises on a missing row; nothing
// maps that to a 404, so an unknown id surfaces as a framework 500.
const unknownId = {
	description: 'The acronym does not exist (the Prisma error is not mapped to a 404).',
};

const pathParams = z.object({
	id: z.string().describe('Acronym id.'),
});

export const openapi: OpenApiRoute = {
	patch: {
		path: pathParams,
		body: {
			description: 'Both fields are optional; non-string values are ignored.',
			schema: z.object({
				name: z.string().optional(),
				description: z.string().optional(),
			}),
		},
		responses: {
			200: { description: 'The updated acronym.', schema: acronymSchema },
			500: unknownId,
		},
	},
	delete: {
		path: pathParams,
		responses: {
			204: { description: 'Deleted. No body.' },
			500: unknownId,
		},
	},
};
