// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { z } from 'zod';
import { errorSchema, promptSchema } from '@/lib/openapiSchemas';
import type { OpenApiRoute } from '@/types/openapi';

// PATCH and DELETE go straight to `prisma.update`/`prisma.delete`, which raise
// on a missing row; nothing translates that into a 404, so an unknown id
// surfaces as a 500 from the framework rather than a structured error body.
const unknownId = {
	description: 'The prompt does not exist (the Prisma error is not mapped to a 404).',
};

const pathParams = z.object({
	id: z.string().describe('Custom prompt id.'),
});

export const openapi: OpenApiRoute = {
	get: {
		path: pathParams,
		responses: {
			200: { description: 'The prompt.', schema: promptSchema },
			404: { description: 'No prompt with that id.', schema: errorSchema },
		},
	},
	patch: {
		path: pathParams,
		body: {
			description: 'Only `content` is writable; a non-string value is ignored.',
			schema: z.object({ content: z.string().optional() }),
		},
		responses: {
			200: { description: 'The updated prompt.', schema: promptSchema },
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
