// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { z } from 'zod';
import type { OpenApiRoute } from '@/types/openapi';

// No request body: the conversation is named in the query string, and the
// backend's own `conversation_id` param is filled in by this proxy from the
// ownership check, so the two spellings never both appear on the wire.
export const openapi: OpenApiRoute = {
	post: {
		query: z.object({
			conversation_id: z
				.string()
				.describe('Conversation whose in-flight run to abort. Must belong to the caller.'),
		}),
		responses: {
			404: {
				description: 'Missing/unknown `conversationId`, or it belongs to another user.',
				contentType: 'text/plain',
				schema: z.literal('Conversation not found'),
			},
			// The backend's 422 is unreachable — this proxy supplies
			// `conversation_id` itself, and any non-2xx upstream is rewritten to
			// the plain-text failure below rather than relayed as a validation body.
			422: null,
			502: {
				description:
					'The backend refused the cancel. The upstream status is used when it has ' +
					'one, 502 otherwise; the body is plain text either way.',
				contentType: 'text/plain',
				schema: z.literal('Upstream cancel request failed'),
			},
		},
	},
};
