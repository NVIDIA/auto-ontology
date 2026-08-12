// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { z } from 'zod';
import { promptSchema } from '@/lib/openapiSchemas';
import type { OpenApiRoute } from '@/types/openapi';

export const openapi: OpenApiRoute = {
	get: {
		responses: {
			200: {
				description: 'Every custom prompt, in no particular order.',
				schema: z.array(promptSchema),
			},
		},
	},
	post: {
		body: {
			description: 'An omitted `content` creates an empty prompt.',
			schema: z.object({ content: z.string().optional() }),
		},
		responses: {
			201: { description: 'The created prompt.', schema: promptSchema },
		},
	},
};
