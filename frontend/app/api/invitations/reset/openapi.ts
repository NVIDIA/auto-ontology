// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { z } from 'zod';
import type { OpenApiRoute } from '@/types/openapi';

export const openapi: OpenApiRoute = {
	post: {
		body: {
			description: 'Id of the existing user who should set a new password via invite link.',
			schema: z.object({
				userId: z.string(),
			}),
		},
		responses: {
			201: {
				description:
					'A reset invite for that user. The account is unchanged until they redeem `url`.',
				schema: z.object({
					url: z.string(),
					email: z.string(),
					expires_at: z.string(),
				}),
			},
			400: { description: '`userId` is missing, or the caller is resetting themselves.' },
			404: { description: 'No user with that id.' },
		},
	},
};
