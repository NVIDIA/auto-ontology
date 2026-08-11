// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { z } from 'zod';
import { enabledFlagSchema } from '@/lib/openapiSchemas';
import type { OpenApiRoute } from '@/types/openapi';

export const openapi: OpenApiRoute = {
	get: {
		responses: {
			200: {
				description:
					'Whether SQL results are auto-visualised. On when unset, so a fresh ' +
					'instance reports `true`.',
				schema: enabledFlagSchema,
			},
		},
	},
	put: {
		body: {
			description: 'Anything other than `true` turns the flag off.',
			schema: z.object({ enabled: z.boolean() }),
		},
		responses: {
			200: { description: 'The stored flag.', schema: enabledFlagSchema },
		},
	},
};
