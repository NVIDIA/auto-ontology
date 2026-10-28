// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { z } from 'zod';
import { enabledFlagSchema } from '@/lib/openapiSchemas';
import type { OpenApiRoute } from '@/types/openapi';

export const unstructuredFlowBody = z.object({ enabled: z.boolean() });

export const openapi: OpenApiRoute = {
	get: {
		responses: {
			200: {
				description: 'Whether chat may answer from uploaded documents. Off when unset.',
				schema: enabledFlagSchema,
			},
		},
	},
	put: {
		body: {
			description: 'Enable or disable document-aware chat.',
			schema: unstructuredFlowBody,
		},
		responses: {
			200: { description: 'The stored flag.', schema: enabledFlagSchema },
		},
	},
};
