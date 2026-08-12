// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { z } from 'zod';
import { errorSchema, messageSchema } from '@/lib/openapiSchemas';
import type { OpenApiRoute } from '@/types/openapi';

const pathParams = z.object({
	id: z.string().describe('Conversation to append the message to. Must belong to the caller.'),
});

export const openapi: OpenApiRoute = {
	post: {
		path: pathParams,
		body: {
			description:
				'Chat turns are written by `POST /api/chat/completions` itself; this route is ' +
				'for callers that persist a turn they produced some other way.',
			schema: z.object({
				role: z.string().describe('`user` or `assistant`.'),
				content: z.string().optional(),
				sql_code: z.string().nullish(),
				sql_response: z.string().nullish(),
			}),
		},
		responses: {
			201: { description: 'The created message.', schema: messageSchema },
			404: {
				description: 'No such conversation, or it belongs to another user.',
				schema: errorSchema,
			},
		},
	},
};
