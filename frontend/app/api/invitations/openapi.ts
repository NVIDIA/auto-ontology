// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { z } from 'zod';
import type { OpenApiRoute } from '@/types/openapi';

const invitationSchema = z.object({
	id: z.string(),
	email: z.string(),
	name: z.string(),
	role: z.enum(['admin', 'viewer']),
	kind: z.enum(['invite', 'reset']),
	status: z.enum(['active', 'expired']),
	expires_at: z.string(),
	created_at: z.string(),
	url: z.string().nullable(),
});

export const openapi: OpenApiRoute = {
	get: {
		responses: {
			200: {
				description:
					'Unused invites (active or expired). Accepted invites are deleted. `url` is set only while active.',
				schema: z.object({ invitations: z.array(invitationSchema) }),
			},
		},
	},
	post: {
		body: {
			description: 'Email-bound invite. No mail is sent; copy `url` and share it.',
			schema: z.object({
				email: z.string(),
				name: z.string().optional(),
				role: z.enum(['admin', 'viewer']),
			}),
		},
		responses: {
			201: {
				description: 'The invite link. Active invites also expose `url` on GET.',
				schema: z.object({
					url: z.string(),
					email: z.string(),
					expires_at: z.string(),
				}),
			},
			400: { description: '`email` or `role` is missing or invalid.' },
			409: { description: 'A user with this email already exists.' },
		},
	},
};
