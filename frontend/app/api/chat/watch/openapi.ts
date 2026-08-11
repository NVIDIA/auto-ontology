// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { z } from 'zod';
import type { OpenApiRoute } from '@/types/openapi';

// Declared here because this handler builds its own upstream request: it
// resolves ownership first, then calls the backend with `conversation_id`.
// Only the SSE response is relayed, so the query param is invisible to the
// merge from `docs/openapi/backend.json`.
export const openapi: OpenApiRoute = {
	get: {
		query: z.object({
			conversation_id: z
				.string()
				.describe(
					'Conversation whose in-flight run to observe. Must belong to the caller.',
				),
		}),
		responses: {
			400: {
				description: 'No `conversation_id` was given.',
				contentType: 'text/plain',
				schema: z.literal('Missing conversation_id'),
			},
			404: {
				description: 'Unknown conversation, or it belongs to another user.',
				contentType: 'text/plain',
				schema: z.literal('Conversation not found'),
			},
			// The backend's 422 is unreachable: this proxy supplies
			// `conversation_id` itself, and a non-2xx upstream is rewritten to the
			// plain-text failure below rather than relayed as a validation body.
			422: null,
			502: {
				description:
					'The backend refused the watch. The upstream status is used when it has ' +
					'one, 502 otherwise; the body is plain text either way.',
				contentType: 'text/plain',
				schema: z.literal('Upstream watch request failed'),
			},
		},
	},
};
