// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
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
					'Whether semantic compilation samples live column values. ' +
					'On when unset, unlike semantic compilation itself.',
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
			200: {
				description:
					'The stored flag. Takes effect from the next compilation run; ' +
					'this does not trigger one.',
				schema: enabledFlagSchema,
			},
		},
	},
};
