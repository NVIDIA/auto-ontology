// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { z } from 'zod';
import { apiTokenSchema } from '@/lib/openapiSchemas';
import type { OpenApiRoute } from '@/types/openapi';

const SESSION_ONLY =
	'These routes require a signed-in session: a token may not mint or revoke tokens, ' +
	'so a leaked one cannot issue itself successors.';

export const openapi: OpenApiRoute = {
	get: {
		sessionOnly: true,
		responses: {
			200: { description: 'The caller’s own API tokens.', schema: z.array(apiTokenSchema) },
			403: { description: SESSION_ONLY },
		},
	},
	post: {
		sessionOnly: true,
		body: {
			description: 'Omit `expires_in_days` for a token that never expires.',
			schema: z.object({
				name: z.string().describe('Label shown in the token list.'),
				expires_in_days: z.number().min(1).max(365).optional(),
			}),
		},
		responses: {
			201: {
				description:
					'The created token. `token` is the plaintext secret and is returned ' +
					'only here — Auto Ontology stores just its hash.',
				schema: apiTokenSchema.extend({ token: z.string() }),
			},
			400: { description: '`name` is missing, or `expires_in_days` is out of range.' },
			403: { description: SESSION_ONLY },
		},
	},
};
