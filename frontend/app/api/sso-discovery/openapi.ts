// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { z } from 'zod';
import { errorSchema } from '@/lib/openapiSchemas';
import type { OpenApiRoute } from '@/types/openapi';

export const openapi: OpenApiRoute = {
	get: {
		query: z.object({
			issuer: z
				.string()
				.describe('HTTPS issuer URL; `/.well-known/openid-configuration` is appended.'),
		}),
		responses: {
			200: {
				description:
					'The endpoints the SSO registration form needs, under their OpenID ' +
					'Connect Discovery names. `discovery_endpoint` is added by this route.',
				schema: z.object({
					issuer: z.string(),
					authorization_endpoint: z.string(),
					token_endpoint: z.string(),
					userinfo_endpoint: z.string().nullable(),
					jwks_uri: z.string().nullable(),
					discovery_endpoint: z.string().describe('The URL that was fetched.'),
				}),
			},
			400: {
				description: '`issuer` is missing, unparseable, or not https.',
				schema: errorSchema,
			},
			502: {
				description:
					'The discovery endpoint answered non-2xx, or its document lacks ' +
					'`issuer`/`authorization_endpoint`/`token_endpoint`.',
				schema: errorSchema,
			},
			504: {
				description: 'The discovery request timed out (8s) or could not be reached.',
				schema: errorSchema,
			},
		},
	},
};
