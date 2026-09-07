// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { z } from 'zod';
import { enabledFlagSchema } from '@/lib/openapiSchemas';
import type { OpenApiRoute } from '@/types/openapi';

// Exported so route.ts enforces exactly what this spec advertises, rather than
// the two drifting apart.
export const columnProfilingBody = z.object({ enabled: z.boolean() });

export const openapi: OpenApiRoute = {
	get: {
		responses: {
			200: {
				description:
					'Whether semantic compilation samples live column values. ' +
					'On when unset, unlike semantic compilation itself.',
				schema: enabledFlagSchema,
			},
		},
	},
	put: {
		body: {
			description:
				'`enabled` must be a boolean. This flag is opt-out, so coercing a ' +
				'malformed value would silently turn profiling off.',
			schema: columnProfilingBody,
		},
		responses: {
			200: {
				description:
					'The stored flag. Takes effect from the next compilation run; ' +
					'this does not trigger one.',
				schema: enabledFlagSchema,
			},
			400: { description: 'Body is not JSON, or `enabled` is not a boolean.' },
		},
	},
};
