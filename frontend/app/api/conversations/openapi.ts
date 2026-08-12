// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { z } from 'zod';
import { conversationSchema } from '@/lib/openapiSchemas';
import type { OpenApiRoute } from '@/types/openapi';

export const openapi: OpenApiRoute = {
	get: {
		responses: {
			200: {
				description: "The caller's own conversations, newest first.",
				schema: z.array(conversationSchema),
			},
		},
	},
	post: {
		body: {
			description: 'An omitted title creates the conversation with an empty one.',
			schema: z.object({ title: z.string().optional() }),
		},
		responses: {
			201: { description: 'The created conversation.', schema: conversationSchema },
		},
	},
};
