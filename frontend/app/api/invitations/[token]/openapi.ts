// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { z } from 'zod';
import type { OpenApiRoute } from '@/types/openapi';

export const openapi: OpenApiRoute = {
	delete: {
		path: z.object({
			token: z.string().describe('Invitation id.'),
		}),
		responses: {
			200: {
				description:
					'The invitation is gone. An unused link stops redeeming; the user account is not deleted.',
				schema: z.object({ id: z.string() }),
			},
			404: { description: 'No invitation with that id.' },
		},
	},
};
