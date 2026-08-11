// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { z } from 'zod';
import type { OpenApiRoute } from '@/types/openapi';

// The handler forwards the parsed body to FastAPI verbatim, so the shape is the
// backend's `ChatRequest`. It is restated here rather than merged from
// `docs/openapi/backend.json` because this hand-rolled streaming proxy also
// enforces conversation permissions before forwarding the body.
export const openapi: OpenApiRoute = {
	post: {
		body: {
			description:
				'Step 1 of a chat turn: SQL plus a formatted answer. Charts are a separate ' +
				'`POST /api/chat/visualize` call. Reuse one `conversation_id` for follow-up ' +
				'turns; an unknown UUID creates a conversation, while an ID owned by another ' +
				'user returns 404.',
			schema: z.object({
				question: z.string().min(1),
				conversation_id: z.string().uuid().nullish(),
				target_db: z.string().nullish(),
				prediction: z.boolean().nullish(),
			}),
		},
		responses: {
			409: {
				description:
					'Relayed from the backend: a run is already in flight for this ' +
					'conversation, or the semantic layer has not been compiled yet.',
				schema: z.object({ detail: z.string() }),
			},
		},
	},
};
