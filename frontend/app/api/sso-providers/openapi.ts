// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { z } from 'zod';
import type { OpenApiRoute } from '@/types/openapi';

export const openapi: OpenApiRoute = {
	get: {
		responses: {
			200: {
				description:
					'Registered SSO providers, secrets excluded — enough for the login page ' +
					'to decide whether to offer "Sign in with SSO".',
				schema: z.object({
					providers: z.array(
						z.object({
							provider_id: z.string(),
							issuer: z.string(),
							domain: z.string(),
						}),
					),
				}),
			},
		},
	},
};
