// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { z } from 'zod';
import type { OpenApiRoute } from '@/types/openapi';

export const openapi: OpenApiRoute = {
	post: {
		body: {
			description: 'Password for the invited email. The account is created on success.',
			schema: z.object({
				password: z.string().min(8).max(128),
			}),
		},
		responses: {
			200: {
				description:
					'Account created and the invitation is deleted. The caller should then sign in with this email.',
				schema: z.object({ email: z.string() }),
			},
			400: { description: 'Password is missing or outside 8–128 characters.' },
			404: { description: 'The invitation is invalid, expired, or already used.' },
			409: { description: 'A user with this email already exists.' },
		},
	},
};
