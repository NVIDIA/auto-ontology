// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { z } from 'zod';
import { conversationSchema, errorSchema, messageSchema } from '@/lib/openapiSchemas';
import type { OpenApiRoute } from '@/types/openapi';

// Every method scopes the lookup by `userId`, so another user's conversation is
// reported as 404 rather than 403.
const notFound = {
	description: 'No such conversation, or it belongs to another user.',
	schema: errorSchema,
};

const pathParams = z.object({
	id: z
		.string()
		.describe('Conversation id. Must belong to the caller; someone else’s id answers 404.'),
});

export const openapi: OpenApiRoute = {
	get: {
		path: pathParams,
		responses: {
			200: {
				description: 'The conversation with its messages in chronological order.',
				schema: conversationSchema.extend({ messages: z.array(messageSchema) }),
			},
			404: notFound,
		},
	},
	patch: {
		path: pathParams,
		body: {
			description: 'Rename the conversation.',
			schema: z.object({ title: z.string().optional() }),
		},
		responses: {
			200: { description: 'The updated conversation.', schema: conversationSchema },
			404: notFound,
		},
	},
	delete: {
		path: pathParams,
		responses: {
			204: { description: 'Deleted. No body.' },
			404: notFound,
		},
	},
};
