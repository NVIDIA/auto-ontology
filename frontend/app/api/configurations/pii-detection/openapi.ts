// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { z } from 'zod';
import { enabledFlagSchema } from '@/lib/openapiSchemas';
import type { OpenApiRoute } from '@/types/openapi';

// Exported so route.ts enforces exactly what this spec advertises, rather than
// the two drifting apart.
export const piiDetectionBody = z.object({ enabled: z.boolean() });

export const openapi: OpenApiRoute = {
	get: {
		responses: {
			200: {
				description:
					'Whether ingest classifies catalog columns as PII. Off when unset, ' +
					'so a fresh instance reports `false`.',
				schema: enabledFlagSchema,
			},
		},
	},
	put: {
		body: {
			description:
				'`enabled` must be a boolean. A malformed value is rejected rather ' +
				'than coerced.',
			schema: piiDetectionBody,
		},
		responses: {
			200: {
				description:
					'The stored flag. Takes effect from the next ingest; this does not ' +
					'trigger one.',
				schema: enabledFlagSchema,
			},
			400: { description: 'Body is not JSON, or `enabled` is not a boolean.' },
		},
	},
};
